function numgrad = computeNumericalGradient(J, theta)
% numgrad = computeNumericalGradient(J, theta)
% theta: a vector of parameters
% J: a function that outputs a real-number. Calling y = J(theta) will return the
% function value at theta. 
  
% Initialize numgrad with zeros
numgrad = zeros(size(theta));

%% ---------- YOUR CODE HERE --------------------------------------
% Instructions: 
% Implement numerical gradient checking, and return the result in numgrad.  
% (See Section 2.3 of the lecture notes.)
% You should write code so that numgrad(i) is (the numerical approximation to) the 
% partial derivative of J with respect to the i-th input argument, evaluated at theta.  
% I.e., numgrad(i) should be the (approximately) the partial derivative of J with 
% respect to theta(i).
%                
% Hint: You will probably want to compute the elements of numgrad one at a time. 

EPSILON = 1e-4;  
  
for i = 1:length(theta)  
    theta_iP = theta;
    theta_iP(i) = theta(i) + EPSILON;
    theta_iM = theta;
    theta_iM(i) = theta(i) - EPSILON;
    JP = J(theta_iP); 
    JM = J(theta_iM); 
    numgrad(i) = (JP - JM )/(2*EPSILON);  
end 



%% ---------------------------------------------------------------
end
