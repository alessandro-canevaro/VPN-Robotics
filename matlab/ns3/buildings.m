clear all

fid = fopen('map_2d.txt','r');
x = zeros(34,34);

ind = 1;
while ~feof(fid)
    xdata=num2cell(str2num(fgets(fid)));
    x(ind,:) = cell2mat(xdata);
    ind=ind+1;
end


mesh(x)
hold on
fclose(fid)