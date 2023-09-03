# LaMIM
Large Medical Image Foundation Model

## For Pre-training
1. Pull the MONAI docker image
`docker pull projectmonai/monai`
2. Run run_docker.sh
3. Run run_pretraining.sh insider docker

## Downsteam tasks fine tuning
1. Download pretrained models ([SSL_ViT_Block16](https://drive.google.com/file/d/1x1VI-0AoMqQZYVcbNoTQxe5ac-t3Ia5R/view?usp=drive_link), [SSL_ViT-Block4](https://drive.google.com/file/d/1ttHL3IeZwuhjLPKS6SeLYjRQW-p6dD1U/view?usp=drive_link)) and place them in the **Pretrained_models** folder.
