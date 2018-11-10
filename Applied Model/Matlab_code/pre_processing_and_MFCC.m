% SIGNAL PREPROCESSING ALGORITHM
%{
Andrés Santiago Arias Páez
David Andrés Rubiano Venegas
%}
%% LEER
fs=44100;
list_dir=dir(fullfile('.\Audios','*.wav'));
name_sound={list_dir.name};
fold= list_dir.folder;
sound = audioread([fold,'\',name_sound{2}])
% Normalize the sound in only one
sound = mean(sound,2);
% Do framing by selecting 25 milliseconds (optimum 0.020 - 0.030)
f_d = 0.025; 
% This function is used to segment the sound.
frames = framing(sound, fs, f_d);
% Calculate the energy of the frames
% Take the dimensions of the rows f and c the columns
[fil,col] = size(frames);
% Vector energia (STE)
energy = 0;
% The sum of the elements squared is stored in a vector in a single row
for i = 1 : fil
    energy(i) = sum(frames(i,:).^2);
end
% Normalize data are between the ranges 0 to 1
energy = energy./max(energy); 
% Rounded frame size.
f_size = round(f_d * fs);
% Join the energies of each frame into a single vector
total_energy = 0;
for j = 1 : length(energy)
    k = length(total_energy);
    total_energy(k : k + f_size) = energy(j);
end
% Remove the rests
id = find(energy >= 0.01);
% Matrix of frames without rests
fr_ws = frames(id,:); 
% Audio reconstruction
data_r = reshape(fr_ws',1,[]);
audiowrite('.\Audios\end.wav',data_r,fs);
sound = audioread(name_sound{1})
sound = mean(sound,2);
[coeffs,delta,deltaDelta,loc] = mfcc(sound,fs,'LogEnergy','ignore');
fid=fopen(['.\','MFCCs','.txt'],'wt');
x=coeffs(1,1:13);
for i=2:size(coeffs,1)
    x=x+coeffs(i,1:13);
end
x=x./length(coeffs);
for j=1:length(x)
    if (j==length(x))
        fprintf(fid,'%6.4f',x(j));
    else
        fprintf(fid,'%6.4f,',x(j));
    end
end


