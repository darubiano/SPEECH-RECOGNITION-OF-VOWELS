% ALGORITMO DE PREPROCESAMIENTO DE SEÑALES 
%{
Andrés Santiago Arias Páez
David Andrés Rubiano Venegas
%}
%% LEER
salida={'.\outmen', '.\outwomen'};
entrada={'.\men', '.\women'};
subfolder={'a','e','i','o','u'};

fs=44100;
for a=1:length(salida)
    if (~exist(salida{a},'dir'))
    mkdir(salida{a});
    end
end
for b=1:length(entrada)
    for c=1:length(subfolder)
    directorio= dir([entrada{b},'\',subfolder{c},'\*.wav']);
        for d=1:length(directorio)
            nombre = directorio(d).name;
            sonido= audioread([entrada{b},'\',subfolder{c},'\',nombre]);

            % normalizar el sonido en uno solo
            sonido = mean(sonido,2);
            % Hacer framing seleccionando 25 milisegundos (optimo 0.020 - 0.030 )
            f_d = 0.025; 
            % Esta funcion es usada para segmentar el sonido.
            frames = framing(sonido, fs, f_d);
            % calcular la energia de los frames
            % Se toma las dimenciones del las filas f y c las columnas
            [fil,col] = size(frames);
            % Vector energia (STE)
            energia = 0;
            % se guarda en un vector la suma de los elementos elevados al cuadrado
            % en una sola fila 
            for i = 1 : fil
                energia(i) = sum(frames(i,:).^2);
            end
            %Normalizar datos esten entre los rangos 0 a 1
            energia = energia./max(energia); 
            % tamaño del frame redondeado.
            f_size = round(f_d * fs);

            % Une las energias de cada frame en un solo vector
            energiatotal = 0;
            for j = 1 : length(energia)
                k = length(energiatotal);
                energiatotal(k : k + f_size) = energia(j);
            end

            % Remover los silencios
            id = find(energia >= 0.01);
            % Matriz de frames sin silencios 
            fr_ws = frames(id,:); 
            % Recostruccion del audio
            data_r = reshape(fr_ws',1,[]);
            if b==1
                name=[salida{b},'\h',nombre];
            else
                name=[salida{b},'\m',nombre];
            end
            audiowrite(name,data_r,fs);
        end
    end  
end 
