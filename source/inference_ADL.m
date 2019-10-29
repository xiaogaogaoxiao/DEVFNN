function [y]=inference_ADL(stream,Weight,nRule,nInput,Center,Spread);

 tau=zeros(nRule,1);
                   for i=1:nRule
      
                       dis=(Center(i,:)-stream(1:nInput));
                       %dis1=dis./Spread(i,:);
                       tau(i)=exp(-0.5*dis*Spread(:,:,i)*dis');
                   end
                   sum_tau=sum(tau);
                   if sum_tau<0.001
                   sum_tau=1;
                   end
                    lambda=tau/sum_tau;
                                    tempat1=zeros(2,nInput);
 %tempat=[0 stream(1:n)];
 for i=1:nInput
 for j=1:2
     if i<=1
     if j<=1
     tempat1(j,i)=stream(i);
     else
     tempat1(j,i)=2*stream(i)*tempat1(j-1,i)-1;    
     end
     else
     if j<=1
 tempat1(j,i)=stream(i);
     else
  tempat1(j,i)=2*stream(i)*tempat1(j-1,i)-tempat1(j,i-1);
     end
     end
 end
 end
 %tempat2=tempat1';
 tempat3=zeros(1,2*nInput);
 tempat3(:)=tempat1;

 
 xek = [1, tempat3]'; 
                    for i=1:nRule,      Psik((i-1)*((2*nInput)+1)+1:i*((2*nInput)+1),1) = lambda(i)*xek;    end
                      
                    y = Psik'*Weight; 


