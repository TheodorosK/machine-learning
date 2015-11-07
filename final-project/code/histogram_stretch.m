function[img_s] = histogram_stretch(img)

old_min = prctile(img, 5);
old_max = prctile(img, 95);

new_min = 0;
new_max = 255;

img_s = (img-old_min) * ((new_max-new_min)/(old_max-old_min)) + new_min;

img_s(img_s < new_min) = new_min;
img_s(img_s > new_max) = new_max;

end