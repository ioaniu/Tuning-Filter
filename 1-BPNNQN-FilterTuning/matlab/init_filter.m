
%% initialize filter
% Initialize the filter characteristics, i.e., start frequency point, end
% frequency point, number of sampling frequency points, desired return loss
% and the pass band (two markers).

filter.start_freq = 740;     % start sampling frequency in MHz
filter.end_freq = 1100;       % end sampling frequency in MHz
filter.N = 401;      % number of sampling points
filter.m1 = 883;     % frequency of the left marker in MHz 
filter.m2 = 956;     % frequency of the right marker in MHz 
filter.threshold = -21;  % design specification for return loss in dB

filter.resolution = (filter.end_freq - filter.start_freq) / (filter.N - 1);
filter.start_point = round((filter.m1 - filter.start_freq) / filter.resolution + 1);
filter.end_point = round((filter.m2 - filter.start_freq) / filter.resolution + 1);

