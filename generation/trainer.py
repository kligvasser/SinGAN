import math
import logging
import models
import os
import torch
import torch.nn.functional as F
from utils.core import imresize
from torch.optim.lr_scheduler import StepLR
from torch.autograd import grad as torch_grad, Variable
from data import get_loader
from utils.recorderx import RecoderX
from utils.misc import save_image_grid, mkdir

class Trainer():
    def __init__(self, args):
        # parameters
        self.args = args
        self.print_model = True
        self.invalidity_margins = None
        self.init_generator = True
        self.parallel = False
        
        if self.args.use_tb:
            self.tb = RecoderX(log_dir=args.save_path)

    def _init_models(self, loader):
        # number of features
        max_features = min(self.args.max_features * pow(2, math.floor(self.scale / 4)), 128)
        min_features = min(self.args.min_features * pow(2, math.floor(self.scale / 4)), 128)

        # initialize model
        if not self.scale or (math.floor(self.scale / 4) != math.floor((self.scale - 1)/ 4)):
            model_config = {'max_features': max_features, 'min_features': min_features, 'num_blocks': self.args.num_blocks, 'kernel_size': self.args.kernel_size, 'padding': self.args.padding}
            d_model = models.__dict__[self.args.dis_model]
            self.d_model = d_model(**model_config)
            self.d_model = self.d_model.to(self.args.device)

        # parallel
        if self.args.device_ids and len(self.args.device_ids) > 1:
            self.d_model = torch.nn.DataParallel(self.d_model, self.args.device_ids)
            self.parallel = True

        # init generator
        if self.init_generator:
            g_model = models.__dict__[self.args.gen_model]
            self.g_model = g_model(**model_config)
            self.g_model = self.g_model.to(self.args.device)
            if self.args.device_ids and len(self.args.device_ids) > 1:
                self.g_model = torch.nn.DataParallel(self.g_model, self.args.device_ids)
            self.g_model.scale_factor = self.args.scale_factor
            self.init_generator = False
            loader.dataset.amps = {'s0': torch.tensor(1.).to(self.args.device)}
        else:
            # add amp
            data = next(iter(loader))
            amps = data['amps']
            reals = data['reals']
            noises = data['noises']
            keys = list(reals.keys())
            next_key = keys[keys.index(self.key) + 1]
            z = self.g_model(reals, amps, noises)
            z = imresize(z.detach(), 1. / self.g_model.scale_factor)
            z = z[:, :, 0:reals[next_key].size(2), 0:reals[next_key].size(3)]
            a = self.args.noise_weight * torch.sqrt(F.mse_loss(z, reals[next_key]))
            loader.dataset.amps.update({next_key: a.to(self.args.device)})

            # add scale
            self.g_model.add_scale(self.args.device)

        # print model
        if self.print_model:
            logging.info(self.g_model)
            logging.info(self.d_model)
            logging.info('Number of parameters in generator: {}'.format(sum([l.nelement() for l in self.g_model.parameters()])))
            logging.info('Number of parameters in discriminator: {}'.format(sum([l.nelement() for l in self.d_model.parameters()])))
            self.print_model = False

        # training mode
        self.g_model.train()
        self.d_model.train()
    
    def _init_eval(self, loader):
        # paramaters 
        self.scale = 0

        # number of features
        max_features = min(self.args.max_features * pow(2, math.floor(self.scale / 4)), 128)
        min_features = min(self.args.min_features * pow(2, math.floor(self.scale / 4)), 128)

        # config
        model_config = {'max_features': max_features, 'min_features': min_features, 'num_blocks': self.args.num_blocks, 'kernel_size': self.args.kernel_size, 'padding': self.args.padding}        

        # init first scale
        g_model = models.__dict__[self.args.gen_model]
        self.g_model = g_model(**model_config)
        self.g_model.scale_factor = self.args.scale_factor

        # add scales
        for self.scale in range(1, self.args.stop_scale + 1):
            self.g_model.add_scale('cpu')

        # load model
        logging.info('Loading model...')
        self.g_model.load_state_dict(torch.load(self.args.model_to_load, map_location='cpu'))
        loader.dataset.amps = torch.load(self.args.amps_to_load, map_location='cpu')

        # cuda
        self.g_model = self.g_model.to(self.args.device)
        for key in loader.dataset.amps.keys():
            loader.dataset.amps[key] = loader.dataset.amps[key].to(self.args.device)

        # print 
        logging.info(self.g_model)
        logging.info('Number of parameters in generator: {}'.format(sum([l.nelement() for l in self.g_model.parameters()])))

        # key
        self.key = 's{}'.format(self.args.stop_scale + 1)

        return loader

    def _init_optim(self):
        # initialize optimizer
        self.g_optimizer = torch.optim.Adam(self.g_model.curr.parameters(), lr=self.args.lr, betas=self.args.gen_betas)
        self.d_optimizer = torch.optim.Adam(self.d_model.parameters(), lr=self.args.lr, betas=self.args.dis_betas)

        # initialize scheduler
        self.g_scheduler = StepLR(self.g_optimizer, step_size=self.args.step_size, gamma=self.args.gamma)
        self.d_scheduler = StepLR(self.d_optimizer, step_size=self.args.step_size, gamma=self.args.gamma)

        # criterion
        self.reconstruction = torch.nn.MSELoss()

    def _init_global(self, loader):
        # adjust scales
        real = loader.dataset.image.clone().to(self.args.device)
        self._adjust_scales(real)

        # set reals
        real = imresize(real, self.args.scale_one)
        loader.dataset.reals = self._set_reals(real)

        # set noises
        loader.dataset.noises = self._set_noises(loader.dataset.reals)

    def _init_local(self, loader):    
        # initialize models
        self._init_models(loader)

        # initialize optimization
        self._init_optim()

        # parameters
        self.losses = {'D': [], 'D_r': [], 'D_gp': [], 'D_f': [], 'G': [], 'G_recon': [], 'G_adv': []}
        self.key = 's{}'.format(self.scale)

    def _adjust_scales(self, image):
        self.args.num_scales = math.ceil((math.log(math.pow(self.args.min_size / (min(image.size(2), image.size(3))), 1), self.args.scale_factor_init))) + 1
        self.args.scale_to_stop = math.ceil(math.log(min([self.args.max_size, max([image.size(2), image.size(3)])]) / max([image.size(2), image.size(3)]), self.args.scale_factor_init))
        self.args.stop_scale = self.args.num_scales - self.args.scale_to_stop

        self.args.scale_one = min(self.args.max_size / max([image.size(2), image.size(3)]), 1)
        image_resized = imresize(image, self.args.scale_one)

        self.args.scale_factor = math.pow(self.args.min_size/(min(image_resized.size(2), image_resized.size(3))), 1 / (self.args.stop_scale))
        self.args.scale_to_stop = math.ceil(math.log(min([self.args.max_size, max([image_resized.size(2), image_resized.size(3)])]) / max([image_resized.size(2), image_resized.size(3)]), self.args.scale_factor_init))
        self.args.stop_scale = self.args.num_scales - self.args.scale_to_stop

    def _set_reals(self, real):
        reals = {}

        # loop over scales
        for i in range(self.args.stop_scale + 1):
            s = math.pow(self.args.scale_factor, self.args.stop_scale - i)
            reals.update({'s{}'.format(i): imresize(real.clone().detach(), s).squeeze(dim=0)})

        return reals

    def _set_noises(self, reals):
        noises = {}

        # loop over scales
        for key in reals.keys():
            noises.update({key: self._generate_noise(reals[key].unsqueeze(dim=0), repeat=(key == 's0')).squeeze(dim=0)})
        
        return noises

    def _generate_noise(self, tensor_like, repeat=False):
        if not repeat:
            noise = torch.randn(tensor_like.size()).to(tensor_like.device)
        else:
            noise = torch.randn((tensor_like.size(0), 1, tensor_like.size(2), tensor_like.size(3)))
            noise = noise.repeat((1, 3, 1, 1)).to(tensor_like.device)

        return noise

    def _save_models(self):
        # save models
        torch.save(self.g_model.state_dict(), os.path.join(self.args.save_path, self.key, '{}_s{}.pt'.format(self.args.gen_model, self.step)))
        torch.save(self.d_model.state_dict(), os.path.join(self.args.save_path, self.key, '{}_s{}.pt'.format(self.args.dis_model, self.step)))

    def _save_last(self, amps):
        # save models
        torch.save(self.g_model.state_dict(), os.path.join(self.args.save_path, '{}.pt'.format(self.args.gen_model)))
        torch.save(self.d_model.state_dict(), os.path.join(self.args.save_path, '{}.pt'.format(self.args.dis_model)))

        # save amps
        torch.save(amps, os.path.join(self.args.save_path, 'amps.pt'))

    def _set_require_grads(self, model, require_grad):
        for p in model.parameters():
            p.requires_grad_(require_grad)

    def _critic_wgan_iteration(self, reals, amps):
        # require grads
        self._set_require_grads(self.d_model, True)

        # get generated data
        generated_data = self.g_model(reals, amps)

        # zero grads
        self.d_optimizer.zero_grad()

        # calculate probabilities on real and generated data
        d_real = self.d_model(reals[self.key])
        d_generated = self.d_model(generated_data.detach())

        # create total loss and optimize
        loss_r = -d_real.mean()
        loss_f = d_generated.mean()
        loss = loss_f + loss_r

        # get gradient penalty
        if self.args.penalty_weight:
            gradient_penalty = self._gradient_penalty(reals[self.key], generated_data)
            loss += gradient_penalty * self.args.penalty_weight

        loss.backward()

        self.d_optimizer.step()

        # record loss
        self.losses['D'].append(loss.data.item())
        self.losses['D_r'].append(loss_r.data.item())
        self.losses['D_f'].append(loss_f.data.item())
        if self.args.penalty_weight:
            self.losses['D_gp'].append(gradient_penalty.data.item())

        # require grads
        self._set_require_grads(self.d_model, False)

        return generated_data

    def _gradient_penalty(self, real_data, generated_data):
        # calculate interpolation
        alpha = torch.rand(real_data.size(0), 1, 1, 1)
        alpha = alpha.expand_as(real_data)
        alpha = alpha.to(self.args.device)
        interpolated = alpha * real_data.data + (1 - alpha) * generated_data.data
        interpolated = Variable(interpolated, requires_grad=True)
        interpolated = interpolated.to(self.args.device)

        # calculate probability of interpolated examples
        prob_interpolated = self.d_model(interpolated)

        # calculate gradients of probabilities with respect to examples
        gradients = torch_grad(outputs=prob_interpolated, inputs=interpolated,
                               grad_outputs=torch.ones(prob_interpolated.size()).to(self.args.device),
                               create_graph=True, retain_graph=True, only_inputs=True)[0]

        # return gradient penalty
        return ((gradients.norm(2, dim=1) - 1) ** 2).mean()

    def _generator_iteration(self, noises, reals, amps, generated_data_adv):
        # zero grads
        self.g_optimizer.zero_grad()

        # get generated data
        generated_data_rec = self.g_model(reals, amps, noises) # reals, amps, noises
        loss = 0.

        # reconstruction loss
        if self.args.reconstruction_weight:
            loss_recon = self.reconstruction(generated_data_rec, reals[self.key])
            loss += loss_recon * self.args.reconstruction_weight
            self.losses['G_recon'].append(loss_recon.data.item())
        
        # adversarial loss
        if self.args.adversarial_weight:
            d_generated = self.d_model(generated_data_adv)
            loss_adv = -d_generated.mean()
            loss += loss_adv * self.args.adversarial_weight
            self.losses['G_adv'].append(loss_adv.data.item())

        # backward loss
        loss.backward()
        self.g_optimizer.step()

        # record loss
        self.losses['G'].append(loss.data.item())

    def _train_iteration(self, loader):
        # set inputs
        data = next(iter(loader))
        noises = data['noises']
        reals = data['reals']
        amps = data['amps']
        
        # critic iteration
        fakes = self._critic_wgan_iteration(reals, amps)

        # only update generator every |critic_iterations| iterations
        if self.step % self.args.num_critic == 0:
            self._generator_iteration(noises, reals, amps, fakes)

        # logging
        if self.step % self.args.print_every == 0:
            line2print = 'Iteration {}'.format(self.step)
            line2print += ', D: {:.6f}, D_r: {:.6f}, D_f: {:.6f}'.format(self.losses['D'][-1], self.losses['D_r'][-1], self.losses['D_f'][-1])
            line2print += ', D_gp: {:.6f}'.format(self.losses['D_gp'][-1])
            line2print += ', G: {:.5f}, G_recon: {:.5f}, G_adv: {:.5f}'.format(self.losses['G'][-1], self.losses['G_recon'][-1], self.losses['G_adv'][-1])
            logging.info(line2print)

        # plots for tensorboard
        if self.args.use_tb:
            if self.args.adversarial_weight:
                self.tb.add_scalar('data/s{}/loss_d'.format(self.scale), self.losses['D'][-1], self.step)
            if self.step > self.args.num_critic:
                self.tb.add_scalar('data/s{}/loss_g'.format(self.scale), self.losses['G'][-1], self.step)

    def _eval_iteration(self, loader):
        # set inputs
        data = next(iter(loader))
        noises = data['noises']
        reals = data['reals']
        amps = data['amps']

        # evaluation
        with torch.no_grad():
            generated_fixed = self.g_model(reals, amps, noises)
            generated_sampled = self.g_model(reals, amps)

        # save image
        self._save_image(generated_fixed, 's{}_fixed.png'.format(self.step))
        self._save_image(generated_sampled, 's{}_sampled.png'.format(self.step))

    def _sample_iteration(self, loader):
        # set inputs
        data_reals = loader.dataset.reals
        reals = {}
        amps = loader.dataset.amps

        # set reals
        for key in data_reals.keys():
           reals.update({key: data_reals[key].clone().unsqueeze(dim=0).repeat(self.args.batch_size, 1, 1, 1)}) 

        # evaluation
        with torch.no_grad():
            generated_sampled = self.g_model(reals, amps)

        # save image
        self._save_image(generated_sampled, 's{}_sampled.png'.format(self.step))

    def _save_image(self, image, image_name):
        image = (image + 1.) / 2.
        directory = os.path.join(self.args.save_path, self.key)
        save_path = os.path.join(directory, image_name)
        mkdir(directory)
        save_image_grid(image.data.cpu(), save_path)

    def _train_single_scale(self, loader):
        # run step iterations
        logging.info('\nScale #{}'.format(self.scale + 1))
        for self.step in range(self.args.num_steps + 1):
            # train
            self._train_iteration(loader)

            # scheduler
            self.g_scheduler.step(epoch=self.step)
            self.d_scheduler.step(epoch=self.step)

            # evaluation
            if (self.step % self.args.eval_every == 0) or (self.step == self.args.num_steps):
                # eval
                self.g_model.eval()
                self._eval_iteration(loader)
                self.g_model.train()

        # sample last
        self.step += 1
        self._sample_iteration(loader)

    def _print_stats(self, loader):
        reals = loader.dataset.reals
        amps = loader.dataset.amps

        logging.info('\nScales:')
        for key in reals.keys():
            logging.info('{}, size: {}x{}, amp: {:.3f}'.format(key, reals[key].size(-2), reals[key].size(-1), amps[key]))

    def train(self):
        # get loader
        loader = get_loader(self.args)
        
        # initialize global
        self._init_global(loader)

        # iterate scales
        for self.scale in range(self.args.stop_scale + 1):
            # initialize local
            self._init_local(loader)
            self._train_single_scale(loader)
            self._save_models()

        # save last
        self._save_last(loader.dataset.amps)

        # print stats
        self._print_stats(loader)

        # close tensorboard
        if self.args.use_tb:
            self.tb.close()

    def eval(self):
        # get loader
        loader = get_loader(self.args)

        # init
        self._init_global(loader)
        loader = self._init_eval(loader)

        # evaluate
        logging.info('Evaluating...')
        for self.step in range(self.args.num_steps):
            self._sample_iteration(loader)

        # close tensorboard
        if self.args.use_tb:
            self.tb.close()