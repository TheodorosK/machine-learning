# https://en.wikipedia.org/wiki/Normalization_(image_processing)
function [vimg_str] = histo_stretch(vimg)

	old_min = prctile(vimg', 5);
	old_max = prctile(vimg', 95);

	new_min = 0
	new_max = 255

	vimg_str = (vimg - old_min) * ((new_max-new_min)/(old_max-old_min)) + new_min;

	vimg_str(vimg_str>new_max) = new_max;
	vimg_str(vimg_str<new_min) = new_min;

end