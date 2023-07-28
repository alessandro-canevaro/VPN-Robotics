clear all
close all
%fid = fopen('datapathloss.txt','r');
fid = fopen('buildings-pathloss-profiler-28-5.txt','r');
%fid = fopen('buildings-pathloss-profiler-16-16.txt','r');
x = zeros(2000,3);

ind = 1;
while ~feof(fid)
    xdata=num2cell(str2num(fgets(fid)));
    xdata = xdata(1:3);
    x(ind,:) = cell2mat(xdata);
    ind=ind+1;
end

x(ind:end,:) = [];
fin = zeros(round(sqrt(size(x,1))),round(sqrt(size(x,1))));

for i = 1:size(x,1)
    fin(x(i,1)+1, x(i,2)+1) = x(i,3); 
end


%surf(fin, 'EdgeColor', 'flat');
%zlim([-120 -80])
%zticks(-120:2:-75)
%colorbar
%hold on

%fid2 = fopen('map_2d_small.txt','r');
fid2 = fopen('map_2d.txt','r');
obst = zeros(34,34);
%obst = zeros(16,16);

ind = 1;
while ~feof(fid2)
    obst_xdata=num2cell(str2num(fgets(fid2)));
    obst(ind,:) = cell2mat(obst_xdata);
    ind=ind+1;
end

obst(obst==0) = NaN;
indices = find(isnan(obst) == 0);
fin(indices) = NaN;

%surf(fin);
surf(fin, 'EdgeColor', 'flat');
grayColor = [.7 .7 .7];
set(gca,'Color',grayColor)
xlim([2 34])
ylim([2 34])
view(0, 90)
%zlim([-120 -80])
%zticks(-120:2:-75)
colorbar
hold on

%figure
surf(obst, 'FaceColor', 'k', 'EdgeAlpha', 0.8);
%surf(obst, 'EdgeAlpha', 0.8);
fclose(fid)
fclose(fid2)