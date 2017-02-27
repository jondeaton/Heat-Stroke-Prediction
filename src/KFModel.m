
% Source: Buller MJ et al. “Estimation of human core temperature from sequential heart rate observations.” Physiol Meas. 2013 Jul;34(7):781-98. doi: 10.1088/0967-3334/34/7/781. Epub 2013 Jun 19.

% This is the MATLAB code provided by Buller et al. that translated into Python for our system


n CT = KFModel(HR,CTstart)

%Inputs:

%HR = A vector of minute to minute HR values.

%CTstart = Core Body Temperature at time 0.

%Outputs:

%CT = A vector of minute to minute CT estimates

%Extended Kalman Filter Parameters

a = 1; gamma = 0.022^2;

b_0 = -7887.1; b_1 = 384.4286; b_2 = -4.5714; sigma = 18.88^2;:wq

%Initialize Kalman filter

x = CTstart; v = 0;%v = 0 assumes confidence with start value.

%Iterate through HR time sequence

for time = 1:length(HR)

    %Time Update Phase

    x_pred = a*x;%Equation 3

    v_pred = (a^2)*v+gamma;%Equation 4

    %Observation Update Phase

    z = HR(time);

    c_vc = 2.*b_2.*x_pred+b_1;%Equation 5

    k = (v_pred.*c_vc)./((c_vc.^2).*v_pred+sigma);%Equation 6

    x = x_pred+k.*(z-(b_2.*(x_pred.^2)+b_1.*x_pred+b_0));%Equation 7

    v = (1-k.*c_vc).*v_pred;%Equation 8

    CT(time) = x;

    end
