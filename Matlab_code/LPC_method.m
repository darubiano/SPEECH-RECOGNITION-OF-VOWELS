%https://la.mathworks.com/help/signal/ref/lpc.html
%{
[sonido,fs] = audioread('a_1_ (7)f.wav');
sonido = mean(sonido,2);
sound(sonido,fs);
[coeffs,g] = lpc (sonido, fs);
%}


%% EXTRACCION DE CARACTERISTICAS 
function MetodoLPC()
    entrada={'.\outmen', '.\outwomen'};
    salida='.\coeficientes';

    fs=44100;

    if (~exist(salida,'dir'))
        mkdir(salida);
    end

    for b=1:length(entrada)
        fid=fopen(['..\coeficientes\','LPCs','.txt'],'wt');
        formato='%6.4f,' ;
        directorio= dir([entrada{b},'\','*.wav']);
        for i=1:length(directorio)
            nombre = directorio(i).name;
            sonido= audioread([entrada{b},'\',nombre]);
            sonido = mean(sonido,2);
            ncoeff=2+fs/1000;
            [coeffs,g] = lpc(sonido, ncoeff);
            fprintf(fid,[nombre,',']);
            for j=1:length(coeffs)
                fprintf(fid,formato,coeffs(j));
            end
            fprintf(fid,[nombre(1),'\n']);
            plot(coeffs)
        end
        fclose(fid);
    end 
end
