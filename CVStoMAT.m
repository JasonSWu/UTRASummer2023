%% Convert all .CSV files to .MAT files
HOME = './';
cd(strcat(HOME))
FileList = dir(fullfile(HOME, '1984-05-02.csv'));
display(FileList)
for iFile = 1:numel(FileList)
    [~, name, ext] = fileparts(FileList(iFile).name);
    data = readtable(fullfile(HOME, [name, ext]));
    save(fullfile(HOME, [name, '.mat']), 'data');
end