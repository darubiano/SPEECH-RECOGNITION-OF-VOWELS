% Esta funcion es usada para segmentar el sonido.
%
function [frames] = framing(x,fs,f_d)

% x: es la entrada de sonido
% fs: la frecuencia del sonido (44100 Hz)
% f_d: duracion de la seleccion (en milisegundos)
% frames: retorna una matriz donde cada fila representa una duracion del
% frame seleccionado

f_size = round(f_d * fs);  % tamaño del frame redondeado.
l_s = length(x);    % longitud del audio
%segmentos del audio dentro de los 25 miliseguntos
n_f = floor(l_s/f_size); % redondeado minimo (trucar).

% crear frames
temp = 0;
% crea la matriz de n_f filas y f_size columnas.
for i = 1 : n_f
    frames(i,:) = x(temp + 1 : temp + f_size);
    temp = temp + f_size;
end

end