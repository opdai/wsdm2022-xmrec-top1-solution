result_pair = [
	('t1_s0', 't2_s0'),
	('t1_s1', 't2_s1'),
	('t1_s1s2', 't2_s1s2'),
	('t1_s1s2s3', 't2_s1s2s3'),
	('t1_s1s3', 't2_s1s3'),
	('t1_s2', 't2_s2'),
	('t1_s2s3', 't2_s2s3'),
	('t1_s3', 't2_s3'),
	('t1_s0__main', 't2_s0__main'),
]

data_combination = {
	't1_s0': {
		'train_batch_size': 4560,
		'train_epoch': 400,
		'latent_dim_rec': 2048,
	},
	't1_s0_use_valid': {
		'train_batch_size': 4560,
		'train_epoch': 400,
		'latent_dim_rec': 2048,
	},
	't1_s1': {
		'train_batch_size': 4560,
		'train_epoch': 220,
		'latent_dim_rec': 1280,
	},
	't1_s1_use_valid': {
		'train_batch_size': 4560,
		'train_epoch': 220,
		'latent_dim_rec': 1280,
	},
	't1_s1s2': {
		'train_batch_size': 4560,
		'train_epoch': 160,
		'latent_dim_rec': 1280,
	},
	't1_s1s2_use_valid': {
		'train_batch_size': 4560,
		'train_epoch': 160,
		'latent_dim_rec': 1280,
	},
	't1_s1s2s3': {
		'train_batch_size': 4560,
		'train_epoch': 120,
		'latent_dim_rec': 1280,
	},
	't1_s1s2s3_use_valid': {
		'train_batch_size': 4560,
		'train_epoch': 120,
		'latent_dim_rec': 1280,
	},
	't1_s1s3': {
		'train_batch_size': 4560,
		'train_epoch': 240,
		'latent_dim_rec': 1280,
	},
	't1_s1s3_use_valid': {
		'train_batch_size': 4560,
		'train_epoch': 240,
		'latent_dim_rec': 1280,
	},
	't1_s2': {
		'train_batch_size': 4096,
		'train_epoch': 300,
		'latent_dim_rec': 2048,
	},
	't1_s2_use_valid': {
		'train_batch_size': 4096,
		'train_epoch': 300,
		'latent_dim_rec': 2048,
	},
	't1_s2s3': {
		'train_batch_size': 4096,
		'train_epoch': 260,
		'latent_dim_rec': 2048,
	},
	't1_s2s3_use_valid': {
		'train_batch_size': 4096,
		'train_epoch': 260,
		'latent_dim_rec': 2048,
	},
	't1_s3': {
		'train_batch_size': 4096,
		'train_epoch': 420,
		'latent_dim_rec': 2048,
	},
	't1_s3_use_valid': {
		'train_batch_size': 4096,
		'train_epoch': 420,
		'latent_dim_rec': 2048,
	},
	't2_s0': {
		'train_batch_size': 4560,
		'train_epoch': 400,
		'latent_dim_rec': 2048,
	},
	't2_s0_use_valid': {
		'train_batch_size': 4560,
		'train_epoch': 400,
		'latent_dim_rec': 2048,
	},
	't2_s1': {
		'train_batch_size': 4560,
		'train_epoch': 180,
		'latent_dim_rec': 1280,
	},
	't2_s1_use_valid': {
		'train_batch_size': 4560,
		'train_epoch': 180,
		'latent_dim_rec': 1280,
	},
	't2_s1s2': {
		'train_batch_size': 4560,
		'train_epoch': 140,
		'latent_dim_rec': 1280,
	},
	't2_s1s2_use_valid': {
		'train_batch_size': 4560,
		'train_epoch': 140,
		'latent_dim_rec': 1280,
	},
	't2_s1s2s3': {
		'train_batch_size': 4560,
		'train_epoch': 120,
		'latent_dim_rec': 1280,
	},
	't2_s1s2s3_use_valid': {
		'train_batch_size': 4560,
		'train_epoch': 120,
		'latent_dim_rec': 1280,
	},
	't2_s1s3': {
		'train_batch_size': 4560,
		'train_epoch': 260,
		'latent_dim_rec': 1280,
	},
	't2_s1s3_use_valid': {
		'train_batch_size': 4560,
		'train_epoch': 260,
		'latent_dim_rec': 1280,
	},
	't2_s2': {
		'train_batch_size': 4096,
		'train_epoch': 260,
		'latent_dim_rec': 2048,
	},
	't2_s2_use_valid': {
		'train_batch_size': 4096,
		'train_epoch': 260,
		'latent_dim_rec': 2048,
	},
	't2_s2s3': {
		'train_batch_size': 4096,
		'train_epoch': 240,
		'latent_dim_rec': 2048,
	},
	't2_s2s3_use_valid': {
		'train_batch_size': 4096,
		'train_epoch': 240,
		'latent_dim_rec': 2048,
	},
	't2_s3': {
		'train_batch_size': 4096,
		'train_epoch': 340,
		'latent_dim_rec': 2048,
	},
	't2_s3_use_valid': {
		'train_batch_size': 4096,
		'train_epoch': 340,
		'latent_dim_rec': 2048,
	},
	't1_s0__main': {
		'train_batch_size': 8192,
		'latent_dim_rec': 2048,
		'train_epoch': 350,
	},
	't2_s0__main': {
		'train_batch_size': 8192,
		'latent_dim_rec': 2048,
		'train_epoch': 250,
	},
	't1_s0__main_use_valid': {
		'train_batch_size': 8192,
		'latent_dim_rec': 2048,
		'train_epoch': 300,
	},
	't2_s0__main_use_valid': {
		'train_batch_size': 8192,
		'latent_dim_rec': 2048,
		'train_epoch': 230,
	}
}
