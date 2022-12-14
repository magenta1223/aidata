python test_single_img_integrated.py \
--img demo/58.png \
--Dconfig configs/denoise/GaussianColorDenoising_Restormer.yml \
--Dweights configs/deraining/deblur.pth \
--Sconfig configs/segmentation/pspnet_r50-d8_512x1024_40k_cityscapes.py \
--Sweights configs/segmentation/pspnet_r50-d8_512x1024_40k_cityscapes_20200605_003338-2966598c.pth \
--device cuda:0 \
--result_dir results \
--out_file result.jpg