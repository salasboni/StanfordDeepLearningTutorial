function [cost,grad] = sparseAutoencoderCost(theta, visibleSize, hiddenSize, ...
                                             lambda, sparsityParam, beta, data)

% visibleSize: the number of input units (probably 64) 
% hiddenSize: the number of hidden units (probably 25) 
% lambda: weight decay parameter
% rho: The desired average activation for the hidden units (denoted in the lecture
%                           notes by the greek alphabet rhohat, which looks like a lower-case "p").
% beta: weight of sparsity penalty term
% data: Our 64x10000 matrix containing the training data.  So, data(:,i) is the i-th training example. 
  
% The input theta is a vector (because minFunc expects the parameters to be a vector). 
% We first convert theta to the (W1, W2, b1, b2) matrix/vector format, so that this 
% follows the notation convention of the lecture notes. 

W1 = reshape(theta(1:hiddenSize*visibleSize), hiddenSize, visibleSize);  
W2 = reshape(theta(hiddenSize*visibleSize+1:2*hiddenSize*visibleSize), visibleSize, hiddenSize);  
b1 = theta(2*hiddenSize*visibleSize+1:2*hiddenSize*visibleSize+hiddenSize);  
b2 = theta(2*hiddenSize*visibleSize+hiddenSize+1:end);  
  
% Cost and gradient variables (your code needs to compute these values).   
% Here, we initialize them to zeros.   

cost = 0;  
W1grad = zeros(size(W1));  
W2grad = zeros(size(W2));  
b1grad = zeros(size(b1));  
b2grad = zeros(size(b2));  
  
%% ---------- YOUR CODE HERE --------------------------------------  
%  Instructions: Compute the cost/optimization objective J_sparse(W,b) for the Sparse Autoencoder,  
%                and the corresponding gradients W1grad, W2grad, b1grad, b2grad.  
%  
% W1grad, W2grad, b1grad and b2grad should be computed using backpropagation.  
% Note that W1grad has the same dimensions as W1, b1grad has the same dimensions  
% as b1, etc.  Your code should set W1grad to be the partial derivative of J_sparse(W,b) with  
% respect to W1.  I.e., W1grad(i,j) should be the partial derivative of J_sparse(W,b)   
% with respect to the input parameter W1(i,j).  Thus, W1grad should be equal to the term   
% [(1/m) \Delta W^{(1)} + \lambda W^{(1)}] in the last block of pseudo-code in Section 2.2   
% of the lecture notes (and similarly for W2grad, b1grad, b2grad).  
%   
% Stated differently, if we were using batch gradient descent to optimize the parameters,  
% the gradient descent update to W1 would be W1 := W1 - alpha * W1grad, and similarly for W2, b1, b2.   
%   
  
JWb = 0;            %squared-error cost  
JWbSparse = 0;      %overall cost function:JWbSparse = JWb + beta*KL  
[n,m] = size(data); %m:sample number  
  
%layer 2
z2 = bsxfun(@plus,W1*data,b1);
a2 = sigmoid(z2);

%layer 3  

z3 = bsxfun(@plus,W2*a2,b2); 
a3 = sigmoid(z3);  
  
%cost function J(W,b)

sumofsquares = (0.5/m)*(sum(sum((a3-data).^2)));  
JWb = sumofsquares + 0.5*lambda*(sum(sum(W1.^2)) + sum(sum(W2.^2)));  
  
%sparsity parameter  
%average activation in layer 2  
rho = sparsityParam;
rhohat = (1/m)*(sum(a2,2));  
KL = getKL(rho,rhohat);

JWbSparse = JWb + beta*sum(KL);  
cost = JWbSparse;  
  
%derivative computation (backpropagation)  
delta3 = -(data-a3) .* sigmoidDeriv(z3);  
sterm = beta .* ((-rho)./rhohat + (1-rho)./(1-rhohat));  
delta2 = (W2'*delta3 + repmat(sterm,1,m)) .* sigmoidDeriv(z2);  
  
% W1grad should be equal to the term [(1/m) \Delta W^{(1)} + \lambda W^{(1)}]  
DeltaW2 = delta3*a2';  
W2grad = DeltaW2./m + lambda * W2;  
DeltaW1 = delta2*data';  
W1grad = DeltaW1./m + lambda * W1;  
  
Deltab2 = sum(delta3,2);  
b2grad = Deltab2./m;  
Deltab1 = sum(delta2,2);  
b1grad = Deltab1./m;  
  
  
%-------------------------------------------------------------------  
% After computing the cost and gradient, we will convert the gradients back  
% to a vector format (suitable for minFunc).  Specifically, we will unroll  
% your gradient matrices into a vector.  
  
grad = [W1grad(:) ; W2grad(:) ; b1grad(:) ; b2grad(:)];  
  
end  
  
%-------------------------------------------------------------------  
% Here's an implementation of the sigmoid function, which you may find useful  
% in your computation of the costs and the gradients.  This inputs a (row or  
% column) vector (say (z1, z2, z3)) and returns (f(z1), f(z2), f(z3)).   
%its derivative is given by f'(z) = f(z)(1 - f(z))  
  
function sigm = sigmoid(x)  
    
    sigm = 1 ./ (1 + exp(-x));  
end  
  
function sigmDeriv = sigmoidDeriv(x)  
  
    sigmDeriv = sigmoid(x).*(1-sigmoid(x));  
end  


function KL = getKL(rho,rhohat)
  
    KL = rho .* log(rho./rhohat) + (1-rho) .* log((1-rho)./(1-rhohat));  

end
%%
