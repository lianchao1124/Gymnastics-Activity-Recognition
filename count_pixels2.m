function [horizontal_count, vertical_count] = count_pixels2(image, pixel_value,iv)
    % Feature extraction and dimensionality reduction
    [rows, cols] = size(image);
    m=1;
    n=1;
    
    % 统计水平方向上各位置的像素点总数
    for col = 1:iv:cols
        horizontal_count(m) = sum(image(:, col) == pixel_value);
        m=m+1;
    end
    
    % 统计垂直方向上各位置的像素点总数
    for row = 1:iv:rows
        vertical_count(n) = sum(image(row, :) == pixel_value);
        n=n+1;
    end
end