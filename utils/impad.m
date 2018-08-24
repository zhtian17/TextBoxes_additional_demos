fileFolder=fullfile('E:\PCBdata\text\org');
dirOutput=dir(fullfile(fileFolder,'*.bmp'));
fileNames={dirOutput.name}';

for i=1:size(fileNames,1)
    name = fileNames{i,1};
    path = fullfile(fileFolder,name);
    img = imread(path);
    res = im2uint8(zeros(500,500,3));
    res(151:350,151:350,:) = img;
    save_path = fullfile('E:\PCBdata\text\padding',name);
    imwrite(res,save_path);
end
