import os
import random
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')

import torch
from torch import nn
from torch.utils.data import DataLoader
try:
	from torch.utils.tensorboard import SummaryWriter
except ImportError:
	class SummaryWriter:
		"""Minimal fallback when the optional tensorboard package is unavailable."""

		def __init__(self, log_dir):
			self.log_dir = log_dir
			os.makedirs(log_dir, exist_ok=True)

		def add_scalar(self, *args, **kwargs):
			return None
import torch.nn.functional as F

# import torch.distributed as dist
# import tempfile
# from torch.nn.parallel import DistributedDataParallel as DDP

from utils import common, train_utils
from criteria import id_loss, w_norm
from configs import data_configs
from datasets.images_dataset import ImagesDataset
from datasets.augmentations import AgeTransformer
from criteria.lpips.lpips import LPIPS
from criteria.aging_loss import AgingLoss
from models.psp import pSp
from training.ranger import Ranger

import collections
import shutil
import numpy as np
import copy
import json
import time

class Coach:
	def __init__(self, opts):
		self.opts = opts

		self.global_step = 0

		self.device = 'cuda'
		self.opts.device = self.device

		# Initialize network
		self.net = pSp(self.opts).to(self.device)
		# make a copy of the original net
		self.net_global = copy.deepcopy(self.net)
		self.net_global.eval()
		for param in self.net_global.parameters():
			param.requires_grad = False
		# show modules of the network
		# print('Network modules:')
		# for name, module in self.net.named_modules():
		# 	print(name)
		# reset encoder/pretrained_encoder in terms of linear layers
  
		# for style in self.net.encoder.styles:
		# 	# each style is a GradualStyleBlock
		# 	# style.linear is a EqualLinear and no reset_parameters() method
		# 	# thus we directly initialize the weights following kaiming_uniform_
		# 	style.linear.weight.data = nn.init.kaiming_uniform_(style.linear.weight.data, a=0.01)
		# 	style.linear.bias.data.zero_()
		# for style in self.net.pretrained_encoder.styles:
		# 	style.linear.weight.data = nn.init.kaiming_uniform_(style.linear.weight.data, a=0.01)
		# 	style.linear.bias.data.zero_()
		# # sanity check so that net and net_global are different after reset
		# print('WARNING: Reset the linear layers of the encoder and pretrained_encoder')
		# assert all([torch.equal(p1, p2) for p1, p2 in zip(self.net.encoder.styles.parameters(), self.net_global.encoder.styles.parameters())]) == False
		# assert all([torch.equal(p1, p2) for p1, p2 in zip(self.net.pretrained_encoder.styles.parameters(), self.net_global.pretrained_encoder.styles.parameters())]) == False

		# Initialize loss
		self.mse_loss = nn.MSELoss().to(self.device).eval()
		if self.opts.lpips_lambda > 0:
			self.lpips_loss = LPIPS(net_type='alex').to(self.device).eval()
		if self.opts.id_lambda > 0:
			self.id_loss = id_loss.IDLoss().to(self.device).eval()
		if self.opts.w_norm_lambda > 0:
			self.w_norm_loss = w_norm.WNormLoss(opts=self.opts)
		if self.opts.aging_lambda > 0:
			self.aging_loss = AgingLoss(self.opts)

		# Initialize optimizer
		self.optimizer = self.configure_optimizers()
		self.optimizer_blender = self.configure_optimizers_blender()
		self.optimizer_decoder = self.configure_optimizers_decoder()
		# The released personalized recipe updates the blender only. Freezing the
		# encoders and decoder avoids retaining several gigabytes of unused grads.
		for param in self.net.parameters():
			param.requires_grad = False
		for param in self.net.blender.parameters():
			param.requires_grad = True

		# Initialize dataset
		self.train_dataset, self.test_dataset = self.configure_datasets()
		self.train_dataloader = DataLoader(self.train_dataset,
										   batch_size=self.opts.batch_size,
										   shuffle=True,
										   num_workers=int(self.opts.workers),
										   drop_last=False)
		self.test_dataloader = DataLoader(self.test_dataset,
										  batch_size=self.opts.test_batch_size,
										  shuffle=False,
										  num_workers=int(self.opts.test_workers),
										  drop_last=True)

		self.age_transformer = AgeTransformer(target_age="uniform_random")

		# personalization
		print(f'Extracting facial features from {self.opts.train_dataset}')
		self.feats_dict = collections.defaultdict(list)
		# store all the training features in a dictionary
		for x, y, x_path, y_path in self.train_dataset:
			# x and y are the same in this case
			x = x.to(self.device).float()
			img = x.unsqueeze(0)
			feat = self.id_loss.extract_feats(img)
			# self.feats_dict[x_path] = feat
			# x_age = self.aging_loss.extract_ages(img) / 100.
			x_age = self.aging_loss.extract_ages_gt(x_path) / 100
			# round to 2 decimal places
			key = round(x_age.item(), 2)
			self.feats_dict[key].append((feat, x_path))
		print(f'Processed all {sum([len(v) for v in self.feats_dict.values()])} training images for personalization')
		# todo: save into opts for future use
		self.train_min_age = int(min(self.feats_dict.keys()) * 100)
		self.train_max_age = int(max(self.feats_dict.keys()) * 100)
		print('Training data min_age (ground truth):', self.train_min_age)
		print('Training data max_age (ground truth):', self.train_max_age)
		assert sum([len(v) for v in self.feats_dict.values()]) == len(self.train_dataset)
		assert 0<= self.train_min_age <= self.train_max_age <= 100, f'age range: {self.train_min_age}, {self.train_max_age}'

		# print('Training face age extraction:')
		# self.age_dict = {}
		# # store all the training ages in a dictionary
		# for x, y, x_path, y_path in self.train_dataset:
		# 	# x and y are the same in this case
		# 	x = x.to(self.device).float()
		# 	img = x.unsqueeze(0)
		# 	age = self.aging_loss.extract_ages(img)
		# 	self.age_dict[x_path] = age.detach().cpu().numpy()
		# print(f'Processed all {len(self.age_dict)} training images for personalization')
		# assert len(self.age_dict) == len(self.train_dataset)
		# min_age = min(self.age_dict.values())
		# max_age = max(self.age_dict.values())
		# print('Training data min_age (estimated):', min_age)
		# print('Training data max_age (estimated):', max_age)
		
		# interpolation mode is False for original SAM loss
		self.interpolation = False
		print('Change AgeTransformer to personalization mode (restricted to the range of the training data)')
		self.age_transformer_interpolation = AgeTransformer(target_age="interpolation", range=(self.train_min_age, self.train_max_age))
		# self.age_transformer = self.age_transformer_interpolation 
		self.age_transformer_extrapolation = AgeTransformer(target_age="extrapolation", range=(self.train_min_age, self.train_max_age))
		
		# no restriction on the range of age transformation
		# print('sanity check for feats_dict: ', self.feats_dict.keys())


		# Initialize logger
		log_dir = os.path.join(opts.exp_dir, 'logs')
		os.makedirs(log_dir, exist_ok=True)
		self.logger = SummaryWriter(log_dir=log_dir)

		# Initialize checkpoint dir
		self.checkpoint_dir = os.path.join(opts.exp_dir, 'checkpoints')
		os.makedirs(self.checkpoint_dir, exist_ok=True)
		self.best_val_loss = None
		if self.opts.save_interval is None:
			self.opts.save_interval = self.opts.max_steps

	def perform_forward_pass(self, x):
		y_hat, latent = self.net.forward(x, return_latents=True)
		return y_hat, latent
	
	def perform_forward_pass_blender(self, x):
		# x: [b, 4, 256, 256]
		# latent_local is the results of w(psp) + delta_w(psp)
		with torch.no_grad():
			latent_local = self.net.encode(x)
			latent_global = self.net_global.encode(x)
		target_ages = x[:, -1, 0, 0]
		# latent_final is the results of w(psp) + delta_w(psp) + delta_w(blender)
		latent_final = self.net.blender(latent_local, latent_global, target_ages, self.global_step)
		y_hat, _ = self.net.decoder(
			[latent_final], 
			input_is_latent=True, 
			randomize_noise=False
			)
		# resize
		y_hat = self.net.face_pool(y_hat)
		latent_local_ = latent_local.clone().detach().requires_grad_(True)
		latent_global_ = latent_global.clone().detach().requires_grad_(True)
		# y_global_ = y_global.clone().detach().requires_grad_(True)
		return y_hat, latent_final, latent_local_, latent_global_

	# def perform_forward_pass_blender(self, x):
	# 	# brute force - mlp(mlp(global / local styles), mlp(target_ages))
	# 	# x: [b, 4, 256, 256]

	# 	_, latent_local = self.net.forward(x, return_latents=True)
	# 	_, latent_global = self.net_global.forward(x, return_latents=True)
	# 	# returned latent styles: [b, 18, 512]
	# 	target_ages = x[:, -1, 0, 0]

	# 	# get latent styles for all ages
	# 	with torch.no_grad():
	# 		latent_global_ages = []
	# 		latent_local_ages = []
	# 		for age in range(0, 101, 10):
	# 			imgs = x[:, 0:3, :, :]
	# 			age = age / 100.
	# 			x_input = [torch.cat((img, age * torch.ones((1, img.shape[1], img.shape[2])).to(self.device)))
	# 						for img in imgs]
	# 			x_input = torch.stack(x_input)
	# 			_, latent_global_age = self.net_global.forward(x_input, return_latents=True)
	# 			_, latent_local_age = self.net.forward(x_input, return_latents=True)
	# 			latent_global_ages.append(latent_global_age)
	# 			latent_local_ages.append(latent_local_age)
	# 		latent_global_ages = torch.stack(latent_global_ages) # [11, b, 18, 512]
	# 		latent_global_ages = latent_global_ages.permute(1, 0, 2, 3) # [b, 11, 18, 512]
	# 		latent_local_ages = torch.stack(latent_local_ages)
	# 		latent_local_ages = latent_local_ages.permute(1, 0, 2, 3)
			
	# 	# latent_final = self.net.blender(latent_local, latent_global, target_ages)
	# 	latent_final = self.net.blender(latent_local, latent_global, latent_local_ages, latent_global_ages, target_ages)
	# 	y_hat, _ = self.net.decoder(
	# 		[latent_final], 
	# 		input_is_latent=True, 
	# 		randomize_noise=False
	# 		)
	# 	# resize
	# 	y_hat = self.net.face_pool(y_hat)
	# 	return y_hat, latent_final
	
	def __set_target_to_source(self, x, input_ages):
		return [torch.cat((img, age * torch.ones((1, img.shape[1], img.shape[2])).to(self.device)))
				for img, age in zip(x, input_ages)]

	def train(self):
		self.net.train()
		training_started = time.monotonic()
		while self.global_step < self.opts.max_steps:
			for batch_idx, batch in enumerate(self.train_dataloader):
				x, y, x_path, y_path = batch
				x, y = x.to(self.device).float(), y.to(self.device).float()

				# input_ages = self.aging_loss.extract_ages(x) / 100.
				input_ages = self.aging_loss.extract_ages_gt(x_path) / 100
				input_ages = input_ages.to(self.device)

				# perform no aging in 33% of the time
				no_aging = random.random() <= (1. / 3)
				if no_aging:
					x_input = self.__set_target_to_source(x=x, input_ages=input_ages)
				else:
					# perform in-domain aging in 50% of the time
					self.interpolation = random.random() <= (1. / 2)
					if self.interpolation:
						x_input = [self.age_transformer_interpolation(img.cpu()).to(self.device) for img in x]
					else:
						x_input = [self.age_transformer_extrapolation(img.cpu()).to(self.device) for img in x]

				x_input = torch.stack(x_input)
				target_ages = x_input[:, -1, 0, 0]

				# print('sanity check:')
				# print('x_input shape:', x_input.shape)
				# x_input shape: torch.Size([b, 4, 256, 256])
				# print('target_ages shape:', target_ages.shape)
				# target_ages shape: torch.Size([b])

				self.encoder_step = 2e4
				# self.encoder_step = 0 # this is to finetune the decoder only
				# self.decoder_step = self.encoder_step + 1e4

				if self.opts.train_encoder:
					# * naive finetuning like SAM
					naive_personalization = False
					if naive_personalization:
						self.optimizer.zero_grad()
						# perform forward/backward pass on real images
						y_hat, latent = self.perform_forward_pass(x_input)
						# todo: temp, remove latent_local / global later
						with torch.no_grad():
							y_global, latent_global = self.net_global.forward(x_input, return_latents=True)
						latent_local, latent_global = latent.clone().detach().requires_grad_(True), latent_global.clone().detach().requires_grad_(True)

						loss, loss_dict, id_logs = self.calc_loss(x, y, y_hat, latent,
																target_ages=target_ages,
																input_ages=input_ages,
																no_aging=no_aging,
																data_type="real",
																latent_local=latent_local,
																latent_global=latent_global,
																)
						
						loss.backward()

						# perform cycle on generate images by setting the target ages to the original input ages
						y_hat_clone = y_hat.clone().detach().requires_grad_(True)
						input_ages_clone = input_ages.clone().detach().requires_grad_(True)
						y_hat_inverse = self.__set_target_to_source(x=y_hat_clone, input_ages=input_ages_clone)
						y_hat_inverse = torch.stack(y_hat_inverse)
						reverse_target_ages = y_hat_inverse[:, -1, 0, 0]
						y_recovered, latent_cycle = self.perform_forward_pass(y_hat_inverse)
						# todo: temp, remove latent_local / global later
						with torch.no_grad():
							y_global_cycle, latent_global_cycle = self.net_global.forward(y_hat_inverse, return_latents=True)
						latent_local_cycle, latent_global_cycle = latent_cycle.clone().detach().requires_grad_(True), latent_global_cycle.clone().detach().requires_grad_(True)
						loss, cycle_loss_dict, cycle_id_logs = self.calc_loss(x, y, y_recovered, latent_cycle,
																			target_ages=reverse_target_ages,
																			input_ages=input_ages,
																			no_aging=no_aging,
																			data_type="cycle",
																			latent_local=latent_local_cycle,
																			latent_global=latent_global_cycle
																			)
						loss.backward()

						# combine the logs of both forwards
						for idx, cycle_log in enumerate(cycle_id_logs):
							id_logs[idx].update(cycle_log)
						loss_dict.update(cycle_loss_dict)
						loss_dict["loss"] = loss_dict["loss_real"] + loss_dict["loss_cycle"]

						self.optimizer.step()

					else:
						self.optimizer_blender.zero_grad()
						# perform forward/backward pass on real images
						# y_hat, latent = self.perform_forward_pass_blender(x_input)
						# y_hat, latent, y_global = self.perform_forward_pass_blender(x_input)
						y_hat, latent, latent_local, latent_global = self.perform_forward_pass_blender(x_input)

						loss, loss_dict, id_logs = self.calc_loss(x, y, y_hat, latent,
																target_ages=target_ages,
																input_ages=input_ages,
																no_aging=no_aging,
																data_type="real",
																latent_local=latent_local,
																latent_global=latent_global,
																)
						
						loss.backward()

						# perform cycle on generate images by setting the target ages to the original input ages
						y_hat_clone = y_hat.clone().detach().requires_grad_(True)
						input_ages_clone = input_ages.clone().detach().requires_grad_(True)
						y_hat_inverse = self.__set_target_to_source(x=y_hat_clone, input_ages=input_ages_clone)
						y_hat_inverse = torch.stack(y_hat_inverse)
						reverse_target_ages = y_hat_inverse[:, -1, 0, 0]
						# y_recovered, latent_cycle, y_global_cycle = self.perform_forward_pass_blender(y_hat_inverse)
						y_recovered, latent_cycle, latent_local_cycle, latent_global_cycle = self.perform_forward_pass_blender(y_hat_inverse)
						loss, cycle_loss_dict, cycle_id_logs = self.calc_loss(x, y, y_recovered, latent_cycle,
																			target_ages=reverse_target_ages,
																			input_ages=input_ages_clone,
																			no_aging=no_aging,
																			data_type="cycle",
																			latent_local=latent_local_cycle,
																			latent_global=latent_global_cycle
																			)
						# y_recovered, latent_cycle = self.perform_forward_pass(y_hat_inverse)
						# loss, cycle_loss_dict, cycle_id_logs = self.calc_loss(x, y, y_recovered, latent_cycle,
						# 													target_ages=reverse_target_ages,
						# 													input_ages=input_ages,
						# 													no_aging=no_aging,
						# 													data_type="cycle",
						# 													latent_local=latent_local,
						# 													latent_global=latent_global
						# 													)
						loss.backward()

						# combine the logs of both forwards
						for idx, cycle_log in enumerate(cycle_id_logs):
							id_logs[idx].update(cycle_log)
						loss_dict.update(cycle_loss_dict)
						loss_dict["loss"] = loss_dict.get("loss_real", 0) + loss_dict.get("loss_cycle", 0)

						self.optimizer_blender.step()

					if self.global_step % self.opts.board_interval == 0:
						if no_aging:
							prefix = 'no_aging'
						elif self.interpolation:
							prefix = 'interpolation'
						else:
							prefix = 'extrapolation'
						self.print_metrics(loss_dict, prefix=prefix)
						self.log_metrics(loss_dict, prefix=prefix)

				if self.opts.train_decoder:
					self.optimizer_decoder.zero_grad()
					# for decoder, solely reconstruction loss purpose
					no_aging = True
					x_input_person = self.__set_target_to_source(x=x, input_ages=input_ages)
					x_input_person = torch.stack(x_input_person)
					input_ages_clone_person = input_ages.clone().detach().requires_grad_(True)
					# perform forward/backward pass on real images
					# y_hat_person, latent_person, *_ = self.perform_forward_pass_blender(x_input_person)
					y_hat_person, latent_person, latent_local_person, latent_global_person = self.perform_forward_pass_blender(x_input_person)

					decoder_loss, decoder_loss_dict, decoder_id_logs = self.calc_loss(
						x=x,
						y=y,
						y_hat=y_hat_person,
						latent=latent_person,
						target_ages=x_input_person[:, -1, 0, 0],
						input_ages=input_ages_clone_person,
						no_aging=no_aging,
						data_type="real",
						latent_local=latent_local_person,
						latent_global=latent_global_person,
						)
					
					loss = decoder_loss

					loss.backward()

					self.optimizer_decoder.step()

					if self.global_step % self.opts.board_interval == 0:
						self.print_metrics(decoder_loss_dict, prefix='decoder')
						self.log_metrics(decoder_loss_dict, prefix='decoder')


				# Logging related
				# if self.global_step % self.opts.image_interval == 0 or \
				# 		(self.global_step < 1000 and self.global_step % 25 == 0):
				# 	self.parse_and_log_images(id_logs, x, y, y_hat, y_recovered,
				# 							  title='images/train/faces')

				# if self.global_step % self.opts.board_interval == 0:
				# 	if self.opts.train_encoder:
				# 		if no_aging:
				# 			prefix = 'no_aging'
				# 		elif self.interpolation:
				# 			prefix = 'interpolation'
				# 		else:
				# 			prefix = 'extrapolation'
				# 		self.print_metrics(loss_dict, prefix=prefix)
				# 		self.log_metrics(loss_dict, prefix=prefix)
				# 	elif self.opts.train_decoder:
				# 		self.print_metrics(decoder_loss_dict, prefix='decoder')
				# 		self.log_metrics(decoder_loss_dict, prefix='decoder')


				self.global_step += 1

				if (self.global_step % self.opts.save_interval == 0
						or self.global_step == self.opts.max_steps):
					self.checkpoint_me(loss_dict, is_best=False)

				self.write_progress(loss_dict, training_started)

				if self.global_step == self.opts.max_steps:
					print('Finished training.', flush=True)
					return

	def checkpoint_me(self, loss_dict, is_best):
		save_name = 'best_model.pt' if is_best else f'iteration_{self.global_step}.pt'
		save_dict = self.__get_save_dict()
		checkpoint_path = os.path.join(self.checkpoint_dir, save_name)
		torch.save(save_dict, checkpoint_path)
		with open(os.path.join(self.checkpoint_dir, 'timestamp.txt'), 'a') as f:
			if is_best:
				f.write('**Best**: Step - {}, '
						'Loss - {:.3f} \n{}\n'.format(self.global_step, self.best_val_loss, loss_dict))
			else:
				f.write(f'Step - {self.global_step}, \n{loss_dict}\n')

	def write_progress(self, loss_dict, training_started):
		state_path = getattr(self.opts, 'state_path', None)
		if not state_path:
			return
		try:
			with open(state_path, 'r', encoding='utf-8') as handle:
				state = json.load(handle)
			elapsed = max(0.0, time.monotonic() - training_started)
			seconds_per_step = elapsed / max(1, self.global_step)
			eta_seconds = seconds_per_step * max(0, self.opts.max_steps - self.global_step)
			state.update({
				'status': 'training',
				'progress': self.global_step,
				'max_steps': self.opts.max_steps,
				'percent': 100.0 * self.global_step / self.opts.max_steps,
				'elapsed_seconds': elapsed,
				'eta_seconds': eta_seconds,
				'latest_metrics': loss_dict,
			})
			temporary = state_path + '.tmp'
			with open(temporary, 'w', encoding='utf-8') as handle:
				json.dump(state, handle, indent=2)
			os.replace(temporary, state_path)
			interval = max(1, getattr(self.opts, 'progress_log_interval', 1))
			if self.global_step == 1 or self.global_step % interval == 0 or self.global_step == self.opts.max_steps:
				loss_value = loss_dict.get('loss', loss_dict.get('loss_real', 0.0))
				print(
					f"Progress {self.global_step}/{self.opts.max_steps} "
					f"({state['percent']:.1f}%) | loss={float(loss_value):.5f} | "
					f"elapsed={elapsed:.0f}s | ETA={eta_seconds:.0f}s",
					flush=True,
				)
		except OSError:
			pass

	# def configure_optimizers(self):
	# 	params = []
	# 	if self.opts.train_encoder:
	# 		params = list(self.net.encoder.parameters())
	# 		for param in self.net.encoder.parameters():
	# 			param.requires_grad = True
	# 	if self.opts.train_decoder:
	# 		params += list(self.net.decoder.parameters())
	# 		for param in self.net.decoder.parameters():
	# 			param.requires_grad = True
	# 	if self.opts.optim_name == 'adam':
	# 		optimizer = torch.optim.Adam(params, lr=self.opts.learning_rate)
	# 	else:
	# 		optimizer = Ranger(params, lr=self.opts.learning_rate)
	# 	return optimizer
 
	# def configure_optimizers(self):
	# 	params = list(self.net.encoder.styles.parameters())
	# 	for name, param in self.net.encoder.named_parameters():
	# 		if 'linear' in name:
	# 			param.requires_grad = True
	# 		else:
	# 			param.requires_grad = False
	# 	params += list(self.net.pretrained_encoder.styles.parameters())
	# 	for name, param in self.net.pretrained_encoder.named_parameters():
	# 		if 'linear' in name:
	# 			param.requires_grad = True
	# 		else:
	# 			param.requires_grad = False
	# 	print('Encoder parameters:')
	# 	for name, param in self.net.encoder.named_parameters():
	# 		print(name, param.requires_grad)
	# 	print('Pretrained encoder parameters:')
	# 	for name, param in self.net.pretrained_encoder.named_parameters():
	# 		print(name, param.requires_grad)
	# 	optimizer = Ranger(params, lr=self.opts.learning_rate)
	# 	return optimizer
 
	def configure_optimizers(self):
		params = list(self.net.encoder.parameters())
		for param in self.net.encoder.parameters():
			param.requires_grad = True
		# params += list(self.net.pretrained_encoder.parameters())
		# for param in self.net.pretrained_encoder.parameters():
		# 	param.requires_grad = True
		optimizer = Ranger(params, lr=self.opts.learning_rate)
		return optimizer

	def configure_optimizers_blender(self):
		params = list(self.net.blender.parameters())
		for param in self.net.blender.parameters():
			param.requires_grad = True
		if self.opts.optim_name == 'adam':
			optimizer = torch.optim.Adam(params, lr=self.opts.learning_rate)
		else:
			optimizer = Ranger(params, lr=self.opts.learning_rate)
		return optimizer
	
	def configure_optimizers_decoder(self):
		params = list(self.net.decoder.parameters())
		for param in self.net.decoder.parameters():
			param.requires_grad = True
		# if self.opts.optim_name == 'adam':
		# 	optimizer = torch.optim.Adam(params, lr=self.opts.learning_rate)
		# else:
		# 	optimizer = Ranger(params, lr=self.opts.learning_rate)
  
		# https://github.com/google/mystyle/blob/f0d5176a9ab9201f7623436b95f8c59a2847b649/reconstruct/tune_net.py#L34
		optimizer = torch.optim.Adam(params, lr=3e-4)
		return optimizer
	
	def configure_datasets(self):
		if self.opts.dataset_type not in data_configs.DATASETS.keys():
			Exception(f'{self.opts.dataset_type} is not a valid dataset_type')
		# print(f'Loading dataset for {self.opts.dataset_type}')
		# print(f'Loading dataset from {self.opts.train_dataset}')
		dataset_args = data_configs.DATASETS[self.opts.dataset_type]
		transforms_dict = dataset_args['transforms'](self.opts).get_transforms()
		if hasattr(self.opts, 'train_dataset') and os.path.isdir(self.opts.train_dataset):
			dataset_args['train_source_root'] = self.opts.train_dataset
			dataset_args['train_target_root'] = self.opts.train_dataset
			print(f'Overwritting training dataset to {self.opts.train_dataset}')
		train_dataset = ImagesDataset(source_root=dataset_args['train_source_root'],
									  target_root=dataset_args['train_target_root'],
									  source_transform=transforms_dict['transform_source'],
									  target_transform=transforms_dict['transform_gt_train'],
									  opts=self.opts)
		test_dataset = ImagesDataset(source_root=dataset_args['test_source_root'],
									 target_root=dataset_args['test_target_root'],
									 source_transform=transforms_dict['transform_source'],
									 target_transform=transforms_dict['transform_test'],
									 opts=self.opts)
		print(f"Number of training samples: {len(train_dataset)}")
		print(f"Number of test samples: {len(test_dataset)}")
		return train_dataset, test_dataset

	def calc_loss(self, x, y, y_hat, latent, target_ages, input_ages, no_aging, data_type="real", latent_local=None, latent_global=None):
		loss = torch.tensor(0.0).to(self.device)
		loss_dict = {}
		id_logs = []
		debug_dir_exp = getattr(self.opts, 'debug_dir', os.path.join(self.opts.exp_dir, 'debug'))
		if os.path.exists(debug_dir_exp) and self.global_step == 0:
			shutil.rmtree(debug_dir_exp)
		os.makedirs(debug_dir_exp, exist_ok=True)
		if self.global_step == 0:
			print('saving debug results to:', debug_dir_exp)

		if self.opts.train_encoder:
			# original sam loss
			if self.opts.id_lambda > 0:
				weights = None
				if self.opts.use_weighted_id_loss:  # compute weighted id loss only on forward pass
					age_diffs = torch.abs(target_ages - input_ages)
					weights = train_utils.compute_cosine_weights(x=age_diffs)
				loss_id, sim_improvement, id_logs = self.id_loss(y_hat, y, x, label=data_type, weights=weights)
				loss_dict[f'loss_id_{data_type}'] = float(loss_id)
				loss_dict[f'id_improve_{data_type}'] = float(sim_improvement)
				loss += loss_id * self.opts.id_lambda
			if self.opts.l2_lambda > 0:
				loss_l2 = F.mse_loss(y_hat, y)
				loss_dict[f'loss_l2_{data_type}'] = float(loss_l2)
				if data_type == "real":
					l2_lambda = self.opts.l2_lambda_aging
				else:
					l2_lambda = self.opts.l2_lambda
				loss += loss_l2 * l2_lambda
			if self.opts.lpips_lambda > 0:
				loss_lpips = self.lpips_loss(y_hat, y)
				loss_dict[f'loss_lpips_{data_type}'] = float(loss_lpips)
				if data_type == "real":
					lpips_lambda = self.opts.lpips_lambda_aging
				else:
					lpips_lambda = self.opts.lpips_lambda
				loss += loss_lpips * lpips_lambda
			if self.opts.lpips_lambda_crop > 0:
				loss_lpips_crop = self.lpips_loss(y_hat[:, :, 35:223, 32:220], y[:, :, 35:223, 32:220])
				loss_dict['loss_lpips_crop'] = float(loss_lpips_crop)
				loss += loss_lpips_crop * self.opts.lpips_lambda_crop
			if self.opts.l2_lambda_crop > 0:
				loss_l2_crop = F.mse_loss(y_hat[:, :, 35:223, 32:220], y[:, :, 35:223, 32:220])
				loss_dict['loss_l2_crop'] = float(loss_l2_crop)
				loss += loss_l2_crop * self.opts.l2_lambda_crop
			if self.opts.w_norm_lambda > 0:
				loss_w_norm = self.w_norm_loss(latent, latent_avg=self.net.latent_avg)
				loss_dict[f'loss_w_norm_{data_type}'] = float(loss_w_norm)
				loss += loss_w_norm * self.opts.w_norm_lambda
			if self.opts.aging_lambda > 0:
				aging_loss, id_logs = self.aging_loss(y_hat, y, target_ages, id_logs, label=data_type)
				loss_dict[f'loss_aging_{data_type}'] = float(aging_loss)
				loss += aging_loss * self.opts.aging_lambda

			# sanity check: save all images
			if no_aging and self.global_step % 50 == 0:
				debug_dir = os.path.join(debug_dir_exp, 'encoder_no_aging')
				os.makedirs(debug_dir, exist_ok=True)
				target_age = target_ages[0].item()
				# save input face
				input_face = common.tensor2im(x[0])
				input_face.save(os.path.join(debug_dir, f'{self.global_step}_input_face_{data_type}.png'))
				# save reconstructed face
				reconstructed = common.tensor2im(y_hat[0])
				reconstructed.save(os.path.join(debug_dir, f'{self.global_step}_loss_{loss.item():.4f}_reconstructed_face_{data_type}_target_age_{target_age}.png'))

			# ---------------------------- interpolation stage --------------------------- #
			if (not no_aging) and self.interpolation:
				# id loss - output vs. self.feats_dict to find the most similar face from the training set
				reconstructed_face_feats = self.id_loss.extract_feats(y_hat) # [b, 512]
				max_sims = []

				# iterate over the batch
				for i, reconstructed_face_feat in enumerate(reconstructed_face_feats):
					# # find the most similar face in the training set using cosine similarity
					# sims = []
					# for feat in self.feats_dict.values():
					# 	sim = F.cosine_similarity(reconstructed_face_feat, feat)
					# 	sims.append(sim)
					# sims = torch.stack(sims)

					# find the most similar face in the training set using cosine similarity
					# but only for the same age
					sims = []
					# input_age = input_ages[i].item()
					# find closest age in the training set
					closest_age = min(self.feats_dict.keys(), key=lambda x: abs(x - target_ages[i].item()))

					# # iterate over the faces with the closest age
					# reference_feats = torch.stack([feat[0] for feat in self.feats_dict[closest_age]])
					# iterate over closest age +- 0.03
					reference_feats = torch.stack([feat[0] for k in self.feats_dict.keys() if abs(k - closest_age) <= 0.03 for feat in self.feats_dict[k]])
					reference_paths = [feat[1] for k in self.feats_dict.keys() if abs(k - closest_age) <= 0.03 for feat in self.feats_dict[k]]
					for feat in reference_feats:
						sim = F.cosine_similarity(reconstructed_face_feat, feat)
						sims.append(sim)
					if len(sims) > 0:
						sims = torch.stack(sims)
						max_sim = torch.max(sims)
						max_sims.append(max_sim)
					else:
						# Handle the case when no reference features are found
						# skip this iteration
						continue

					# sanity check: save all images
					if self.global_step % 50 == 0 and i == 0:
						max_sim_idx = torch.argmax(sims)
						max_sim = sims[max_sim_idx]
						debug_dir = os.path.join(debug_dir_exp, 'encoder_real_aging') if data_type == "real" else os.path.join(debug_dir_exp, 'encoder_cycle_aging')
						os.makedirs(debug_dir, exist_ok=True)

						# save input face
						input_face = common.tensor2im(x[i])
						input_face.save(os.path.join(debug_dir, f'{self.global_step}_input_face_input_age_{input_ages[i]}.png'))
						# save reconstructed face
						reconstructed = common.tensor2im(y_hat[i])
						reconstructed.save(os.path.join(debug_dir, f'{self.global_step}_reconstructed_face_target_age_{target_ages[i]}.png'))
						# save the most similar face
						# most_similar_face_path = list(self.feats_dict.keys())[max_sim_idx]
						# most_similar_face_path = self.feats_dict[closest_age][max_sim_idx][1]
						most_similar_face_path = reference_paths[max_sim_idx]
						from PIL import Image
						most_similar_face = Image.open(most_similar_face_path)
						reference_age = os.path.basename(most_similar_face_path).split('.')[0].split('_')[0].split(' ')[0]
						most_similar_face.save(os.path.join(debug_dir, f'{self.global_step}_ref_age_{reference_age}_sim_{max_sim.item():.4f}.png'))

				max_sims = torch.stack(max_sims)

				nearest_neighbor_id_loss = torch.mean(1 - max_sims)
				# todo: add lambda weights
				loss += nearest_neighbor_id_loss
				loss_dict[f'nearest_neighbor_id_loss'] = float(nearest_neighbor_id_loss)

			# ---------------------------- extrapolation stage --------------------------- #
			with torch.no_grad():
				y_global, _ = self.net_global.decoder([latent_global], input_is_latent=True, randomize_noise=False)
				y_global = self.net_global.face_pool(y_global)
				y_local = y_hat
			if (not no_aging) and (not self.interpolation):
				# sanity check: save all images
				if self.global_step % 50 == 0:
					debug_dir = os.path.join(debug_dir_exp, 'extrapolation')
					os.makedirs(debug_dir, exist_ok=True)
					target_age = target_ages[0].item()
					# save input face
					input_face = common.tensor2im(x[0])
					input_face.save(os.path.join(debug_dir, f'{self.global_step}_{input_ages[0]}_input_face.png'))
					# save reconstructed face
					reconstructed = common.tensor2im(y_hat[0])
					reconstructed.save(os.path.join(debug_dir, f'{self.global_step}_{data_type}_target_age_{target_age:.4f}_reconstructed_face.png'))
					# save the most similar face
					reconstructed_global = common.tensor2im(y_global[0])
					reconstructed_global.save(os.path.join(debug_dir, f'{self.global_step}_{data_type}_target_age_{target_age:.4f}_reconstructed_face_global.png'))

				# ---------------------------------------------------------------------------- #
				#                               global vs. local                               #
				# ---------------------------------------------------------------------------- #
				loss = torch.tensor(0.0).to(self.device)
				extrapolation_loss = torch.tensor(0.0).to(self.device)

				if self.opts.id_lambda > 0:
					weights = None
					if self.opts.use_weighted_id_loss:  # compute weighted id loss only on forward pass
						# calculate age_diffs based on self.train_min_age and self.train_max_age
						age_diffs = []
						for age in target_ages:
							age_diff = min(abs(age - self.train_min_age/100), abs(age - self.train_max_age/100))
							age_diffs.append(age_diff)
						age_diffs = torch.stack(age_diffs)
						weights = train_utils.compute_cosine_weights(x=age_diffs)
					loss_id, sim_improvement, id_logs = self.id_loss(y_hat, y, x, label=data_type, weights=weights)
					loss_dict[f'loss_id_{data_type}'] = float(loss_id)
					loss_dict[f'id_improve_{data_type}'] = float(sim_improvement)
					extrapolation_loss += loss_id * self.opts.id_lambda
				if self.opts.aging_lambda > 0:
					aging_loss, id_logs = self.aging_loss(y_hat, y, target_ages, id_logs, label=data_type)
					loss_dict[f'loss_aging_{data_type}'] = float(aging_loss)
					extrapolation_loss += aging_loss * self.opts.aging_lambda
				# ----------------- above is just part of SAM's original loss ---------------- #
	
				if self.opts.l2_lambda > 0:
					loss_l2 = F.mse_loss(y_hat, y_global)
					loss_dict[f'loss_l2_{data_type}'] = float(loss_l2)
					l2_lambda = self.opts.l2_lambda
					extrapolation_loss += loss_l2 * l2_lambda
				if self.opts.lpips_lambda > 0:
					loss_lpips = self.lpips_loss(y_hat, y_global)
					loss_dict[f'loss_lpips_{data_type}'] = float(loss_lpips)
					lpips_lambda = self.opts.lpips_lambda
					extrapolation_loss += loss_lpips * lpips_lambda
				if self.opts.lpips_lambda_crop > 0:
					loss_lpips_crop = self.lpips_loss(y_hat[:, :, 35:223, 32:220], y_global[:, :, 35:223, 32:220])
					loss_dict['loss_lpips_crop'] = float(loss_lpips_crop)
					extrapolation_loss += loss_lpips_crop * self.opts.lpips_lambda_crop
				if self.opts.l2_lambda_crop > 0:
					loss_l2_crop = F.mse_loss(y_hat[:, :, 35:223, 32:220], y_global[:, :, 35:223, 32:220])
					loss_dict['loss_l2_crop'] = float(loss_l2_crop)
					extrapolation_loss += loss_l2_crop * self.opts.l2_lambda_crop
				if self.opts.aging_lambda > 0:
					aging_loss, id_logs = self.aging_loss(y_hat, y_global, target_ages, id_logs, label=data_type)
					loss_dict[f'loss_aging_{data_type}'] = float(aging_loss)
					extrapolation_loss += aging_loss * self.opts.aging_lambda
	
				if self.opts.w_norm_lambda > 0:
					loss_w_norm = torch.tensor(0.0).to(self.device)
					age_diffs = torch.abs(target_ages - input_ages)
					# weights = (1 - train_utils.compute_cosine_weights(x=age_diffs)) * 1
					# weights = (1 - train_utils.compute_cosine_weights(x=age_diffs)) * 5 if using recon loss
					weights = (1 - train_utils.compute_cosine_weights(x=age_diffs)) * self.opts.adaptive_w_norm_lambda
					# age_diffs = []
					# for age in target_ages:
					# 	age_diff = min(abs(age - self.train_min_age/100), abs(age - self.train_max_age/100))
					# 	age_diffs.append(age_diff)
					# age_diffs = torch.stack(age_diffs)
					# # weights = (1 - train_utils.compute_cosine_weights(x=age_diffs)) * 7
					# weights = (1 - train_utils.compute_cosine_weights(x=age_diffs)) * self.opts.adaptive_w_norm_lambda
					for i in range(len(latent)):
						cur_loss = self.w_norm_loss(latent[i].unsqueeze(0), latent_avg=self.net.latent_avg)
						loss_w_norm += cur_loss * weights[i]
					loss_dict[f'loss_w_norm_{data_type}'] = float(loss_w_norm)
					extrapolation_loss += loss_w_norm * self.opts.w_norm_lambda

				# final loss for extrapolation, do not comment this line
				loss += extrapolation_loss
			loss_dict[f'loss_{data_type}'] = float(loss)
			if data_type == "cycle":
				loss = loss * self.opts.cycle_lambda

			return loss, loss_dict, id_logs
		
		if self.opts.train_decoder:
			# if not no_aging:
			# 	# id loss - output vs. self.feats_dict to find the most similar face from the training set
			# 	reconstructed_face_feats = self.id_loss.extract_feats(y_hat) # [b, 512]
			# 	max_sims = []
			# 	# # todo: reconsider age_diffs between target_ages(y_hat) and training set(self.feats_dict)
			# 	# weights = train_utils.compute_cosine_weights(x=age_diffs)
			# 	# weights = torch.from_numpy(weights).to(self.device)

			# 	# iterate over the batch
			# 	for i, reconstructed_face_feat in enumerate(reconstructed_face_feats):
			# 		# # find the most similar face in the training set using cosine similarity
			# 		# sims = []
			# 		# for feat in self.feats_dict.values():
			# 		# 	sim = F.cosine_similarity(reconstructed_face_feat, feat)
			# 		# 	sims.append(sim)
			# 		# sims = torch.stack(sims)

			# 		# find the most similar face in the training set using cosine similarity
			# 		# but only for the same age
			# 		sims = []
			# 		# input_age = input_ages[i].item()
			# 		# find closest age in the training set
			# 		closest_age = min(self.feats_dict.keys(), key=lambda x: abs(x - target_ages[i].item()))

			# 		# # iterate over the faces with the closest age
			# 		# reference_feats = torch.stack([feat[0] for feat in self.feats_dict[closest_age]])
			# 		# iterate over closest age +- 0.03
			# 		reference_feats = torch.stack([feat[0] for k in self.feats_dict.keys() if abs(k - closest_age) <= 0.03 for feat in self.feats_dict[k]])
			# 		reference_paths = [feat[1] for k in self.feats_dict.keys() if abs(k - closest_age) <= 0.03 for feat in self.feats_dict[k]]
			# 		for feat in reference_feats:
			# 			sim = F.cosine_similarity(reconstructed_face_feat, feat)
			# 			sims.append(sim)
			# 		sims = torch.stack(sims)

			# 		# sanity check: save all images
			# 		if self.global_step % 50 == 0 and i == 0:
			# 			max_sim_idx = torch.argmax(sims)
			# 			max_sim = sims[max_sim_idx]

			# 			debug_dir = os.path.join(debug_dir_exp, 'aging')
			# 			os.makedirs(debug_dir, exist_ok=True)

			# 			# save input face
			# 			input_face = common.tensor2im(x[i])
			# 			input_face.save(os.path.join(debug_dir, f'{self.global_step}_{i}_input_face_input_age_{input_ages[i]}.png'))
			# 			# save reconstructed face
			# 			reconstructed = common.tensor2im(y_hat[i])
			# 			reconstructed.save(os.path.join(debug_dir, f'{self.global_step}_{i}_reconstructed_face_target_age_{target_ages[i]}.png'))
			# 			# save the most similar face
			# 			# most_similar_face_path = list(self.feats_dict.keys())[max_sim_idx]
			# 			# most_similar_face_path = self.feats_dict[closest_age][max_sim_idx][1]
			# 			most_similar_face_path = reference_paths[max_sim_idx]
			# 			from PIL import Image
			# 			most_similar_face = Image.open(most_similar_face_path)
			# 			reference_age = os.path.basename(most_similar_face_path).split('.')[0].split('_')[0].split(' ')[0]
			# 			most_similar_face.save(os.path.join(debug_dir, f'{self.global_step}_{i}_most_similar_face_ref_age_{reference_age}_sim_{max_sim.item():.4f}.png'))

			# 		max_sim = torch.max(sims)
			# 		max_sims.append(max_sim)
			# 	max_sims = torch.stack(max_sims)
			# 	# loss = torch.mean((1 - max_sims) * weights)
			# 	loss = torch.mean(1 - max_sims)
			# 	loss_dict[f'nearest_neighbor_id_loss'] = float(loss)

			# if no_aging:
			# l2 loss
			loss_l2 = F.mse_loss(y_hat, y)
			loss_dict[f'decoder_l2'] = loss_l2.item()

			# lpips loss
			loss_lpips = self.lpips_loss(y_hat, y)
			loss_dict[f'decoder_lpips'] = loss_lpips.item()

			loss += loss_l2 + loss_lpips

			# sanity check: save all images
			if self.global_step % 50 == 0:

				debug_dir = os.path.join(debug_dir_exp, 'decoder_reconstruction')
				os.makedirs(debug_dir, exist_ok=True)
				
				input_face = common.tensor2im(y[0])
				input_face.save(os.path.join(debug_dir, f'{self.global_step}_input_face_input_age_{input_ages[0]:.4f}.png'))
				reconstructed = common.tensor2im(y_hat[0])
				reconstructed.save(os.path.join(debug_dir, f'{self.global_step}_reconstructed_face.png'))

			return loss, loss_dict, id_logs

	def log_metrics(self, metrics_dict, prefix):
		for key, value in metrics_dict.items():
			self.logger.add_scalar(f'{prefix}/{key}', value, self.global_step)

	def print_metrics(self, metrics_dict, prefix):
		print(f'Metrics for {prefix}, step {self.global_step}')
		for key, value in metrics_dict.items():
			print(f'\t{key} = ', value)

	def parse_and_log_images(self, id_logs, x, y, y_hat, y_recovered, title, subscript=None, display_count=2):
		im_data = []
		for i in range(display_count):
			cur_im_data = {
				'input_face': common.tensor2im(x[i]),
				'target_face': common.tensor2im(y[i]),
				'output_face': common.tensor2im(y_hat[i]),
				'recovered_face': common.tensor2im(y_recovered[i])
			}
			if id_logs is not None:
				for key in id_logs[i]:
					cur_im_data[key] = id_logs[i][key]
			im_data.append(cur_im_data)
		self.log_images(title, im_data=im_data, subscript=subscript)

	def log_images(self, name, im_data, subscript=None, log_latest=False):
		fig = common.vis_faces(im_data)
		step = self.global_step
		if log_latest:
			step = 0
		if subscript:
			path = os.path.join(self.logger.log_dir, name, '{}_{:04d}.jpg'.format(subscript, step))
		else:
			path = os.path.join(self.logger.log_dir, name, '{:04d}.jpg'.format(step))
		os.makedirs(os.path.dirname(path), exist_ok=True)
		fig.savefig(path)
		plt.close(fig)

	def __get_save_dict(self):
		save_dict = {
			'state_dict': self.net.state_dict(),
			'opts': vars(self.opts)
		}
		# save the latent avg in state_dict for inference if truncation of w was used during training
		if self.net.latent_avg is not None:
			save_dict['latent_avg'] = self.net.latent_avg
		return save_dict
