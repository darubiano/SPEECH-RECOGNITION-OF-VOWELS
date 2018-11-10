% This function is used to segment the sound.
%
function [frames] = framing(x,fs,f_d)

% x: is the sound input
% fs: the sound frequency (44100 Hz)
% f_d: duration of the selection (in milliseconds)
% frames: returns a matrix where each row represents a duration of
% Selected frame

f_size = round(f_d * fs);  % Rounded frame size.
l_s = length(x);    % Audio length
% Audio segments within 25 milliseconds
n_f = floor(l_s/f_size); % Minimal rounded (trick).

% create frames
temp = 0;
% create the matrix of n_f rows and f_size columns.
for i = 1 : n_f
    frames(i,:) = x(temp + 1 : temp + f_size);
    temp = temp + f_size;
end

end