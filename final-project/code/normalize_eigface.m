% ref: http://bytefish.de/blog/eigenfaces/

function[norm_eigface] = normalize_eigface(eigface)

old_min = min(eigface);
old_max = max(eigface);

% normalize between [0...1]
norm_eigface = eigface - old_min;
norm_eigface = norm_eigface ./ (old_max - old_min);

% scale between [0 255]
norm_eigface = norm_eigface .* (255-0);

end