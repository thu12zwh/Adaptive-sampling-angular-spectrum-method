% please refer to the OL paper titled "Adaptive-sampling angular spectrum method with
%full utilization of space-bandwidth product" 

clc
clear all
close all

iflag = -1;
eps = 10^(-12);
%% input parameters
n0 = 1024; r = 2*n0/5;
t = zeros(n0,1);
t (n0/2-r+1/2:n0/2+r-1/2,1) = 1;
lam = 500e-6;
k = 2*pi/lam;
n = size(t,1);
pitch = 0.001;
l = n*pitch;
x = linspace(-l/2,l/2-l/n,n)';
fx = linspace(-1/2/pitch,1/2/pitch-1/n/pitch,n)';
figure,plot(x,t);title('object')
t_FT = fftshift(fft(fftshift(t)));

zc = n*pitch^2/lam;
z = 50;

%% analytical integral 
X = x;
uu = zeros(n,1);
tic
for j = 1:n
      fun = @(xn) 1/2/pi*z./sqrt((X(j)-xn).^2+z^2).*(1./sqrt((X(j)-xn).^2+z^2)...
               -1i*k/pi).*exp(1i*k*sqrt((X(j)-xn).^2+z^2))./sqrt((X(j)-xn).^2+z^2);
    uu(j,1) = integral(fun,-(r+1/2)*pitch,(r-1/2)*pitch);
end
toc
uu = uu/max(abs(uu));
amplitude_rsi = abs(uu);
phase_rsi = angle(uu);
figure,plot(X,amplitude_rsi);title('Analytical integral amplitude')
set(gca,'looseInset',[0 0 0 0])


K = n/2/max(abs(fx));
%% adaptive sampling number (case I)
fc1 = n*pitch/lam/z;
if z <=zc*2
    nn1 = 2*n;
    fxn1 = linspace(-1/2/pitch,1/2/pitch-1/nn1/pitch,nn1)';
else
    nn1 = round(4*n^2*pitch^2/lam/z);
    fxn1 = linspace(-fc1,fc1-2*fc1/nn1,nn1)';
end

Hn1 = exp(1i*k*z*sqrt(1-(fxn1*lam).^2));

tic
t_1 = nufft1d3(n,x/max(abs(x))*pi,t,iflag,eps,nn1,(fxn1)*K);
t_pro_1 = nufft1d3(nn1,(fxn1)*K,Hn1.*t_1,-iflag,eps,n,x/(max(abs(x)))*pi);
toc

t_pro_1 = t_pro_1/max(abs(t_pro_1));
amplitude_asm_1 = abs(t_pro_1);
phase_asm_1 = angle(t_pro_1);
figure,plot(x,amplitude_asm_1);title('Case I amplitude')
set(gca,'looseInset',[0 0 0 0])
%% adaptive sampling pitch (case II)
fc2 = sqrt(n/2/lam/z);

nn2 = 2*n;

if z <=zc*2
    fxn2 = linspace(-1/2/pitch,1/2/pitch-1/nn2/pitch,nn2)';
else
    fxn2 = linspace(-fc2,fc2-2*fc2/nn2,nn2)';
end

Hn2 = exp(1i*k*z*sqrt(1-(fxn2*lam).^2));

tic
t_2 = nufft1d3(n,x/max(abs(x))*pi,t,iflag,eps,nn2,(fxn2)*K);
t_pro_2 = nufft1d3(nn2,(fxn2)*K,Hn2.*t_2,-iflag,eps,n,x/(max(abs(x)))*pi);
toc

t_pro_2 = t_pro_2/max(abs(t_pro_2));
amplitude_asm_2 = abs(t_pro_2);
phase_asm_2 = angle(t_pro_2);
figure,plot(x,amplitude_asm_2);title('Case II amplitude')
set(gca,'looseInset',[0 0 0 0])


%% SNR
alpha_num = (sum(t_pro_1.*conj(uu)))/(sum((abs(uu)).^2));
snr_num = 10*log10((sum((abs(t_pro_1)).^2))/(sum((abs(t_pro_1-alpha_num*uu)).^2)))

alpha_pitch = (sum(t_pro_2.*conj(uu)))/(sum((abs(uu)).^2));
snr_pitch = 10*log10(sum((abs(t_pro_2)).^2)/sum((abs(t_pro_2-alpha_pitch*uu)).^2))

