
% Define inputs for the test
x = 0; y = 1.5; d = 5; dip = 90; L = 3; W = 2;
z=0; strike=0;
G=30;
nu=0.25;
slip=1;
% Run the MATLAB version of okada92

[displacement1,strain1] = computeOkada92(slip,x,y,z,G,nu,d,dip,L,W,'S',0,strike);

disp = squeeze(displacement1)*1e3

grads = squeeze(strain1)