path = 'div8k';
loadpath = strcat(path, '/raw/trainHR/*.png');
files = dir(loadpath);
scales = [2,4,8,16];
for file = files'
    I = imread(strcat(file.folder,'/',file.name));
    name = split(file.name,'.');
    name = name{1};
    for scale = scales
        J = imresize(I, 1/scale);
        if scale == 16
            writedir = strcat(path,'/raw','/trainLR');
        elseif scale == 8
            writedir = strcat(path,'/raw','/trainLOD1');
        elseif scale == 4
            writedir = strcat(path,'/raw','/trainLOD2');
        elseif scale == 2
            writedir = strcat(path,'/raw','/trainLOD3');
        end
        % mkdir(writedir)
        writepath = strcat(writedir,'/',name,'.png');
        imwrite(J, writepath, 'png');
  end
end
