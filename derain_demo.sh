python test_single_img_integrated.py \
--img demo/58.png \
--Dconfig configs/derain/Deraining_Restormer.yml \
--Dweights configs/derain/derain.pth \
--Sconfig configs/segmentation/pspnet/pspnet_r50-d8_512x1024_40k_cityscapes.py \
--Sweights configs/segmentation/pspnet/pspnet_r50-d8_512x1024_40k_cityscapes_20200605_003338-2966598c.pth \
--device cuda:0 \
--result_dir results \
--out_file result.jpg