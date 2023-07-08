% Sample matrices
matrix1 = readmatrix("./NonInfilled/1984-05-02.csv");
matrix2 = readmatrix("./Infilled/1984-09-23.csv");

% Create a figure with two subplots
figure('Position', [100, 100, 1200, 400]); % [left, bottom, width, height];
subplot(1, 2, 1); % First subplot
imagesc(matrix1);
clim([-20, 20])
colorbar; % Optional: Add a colorbar
title('Matrix 1');

subplot(1, 2, 2); % Second subplot
imagesc(matrix2);
clim([-20, 20])
colorbar; % Optional: Add a colorbar
title('Matrix 2');