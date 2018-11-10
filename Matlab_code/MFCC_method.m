%https://la.mathworks.com/help/audio/ref/mfcc.html#mw_57b2e285-7eab-4759-bcad-816fc821fdc3
% ALGORITMO MEL
%{
Andrés Santiago Arias Páez
David Andrés Rubiano Venegas
%}
%% EXTRACCION DE CARACTERISTICAS 
entrada={'.\outmen', '.\outwomen'};
salida='.\coeficientes';
fs=44100;

if (~exist(salida,'dir'))
    mkdir(salida);
end

fid=fopen(['..\coeficientes\','MFCCs','.txt'],'wt');
formato='%6.4f,' ;
for b=1:length(entrada)
    directorio= dir([entrada{b},'\','*.wav']);
    for i=1:length(directorio)
        nombre = directorio(i).name;
        sonido= audioread([entrada{b},'\',nombre]);
        sonido = mean(sonido,2);
        [coeffs,delta,deltaDelta,loc] = mfcc(sonido,fs,'LogEnergy','ignore');
        disp(nombre)
        x=coeffs(1,1:13);
        for i=2:size(coeffs,1)
            x=x+coeffs(i,1:13);
        end
        x=x./length(coeffs);
        fprintf(fid,[nombre,',']);
        for j=1:length(x)
            fprintf(fid,formato,x(j));
        end
        fprintf(fid,[nombre(2),'\n']);
    end
end
fclose(fid);
plot(coeffs)
