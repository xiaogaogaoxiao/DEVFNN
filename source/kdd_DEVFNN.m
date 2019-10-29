clc
nFolds=1000;
ninput=41;
noutput=2;
subset=3;
decreasingfactor=0.01;
kprune=9*10^(-1);
kfs=0.01;
vigilance=9.*10^(-1);
threshold=0.1;
threshold1=0.8;
merging=1;
% lambdaD=0.001;
% lambdaW=0.005;
drift=1; %choose one to activate the local forgetting mechanism
type_feature_weighting=6; %select 6 to choose the FSC in the empirical feature space
sample_deletion=1;% select one to activate the online active learning scenario
input_selection=0; %select one to activate the feature weighting mechanism
initial=0.5;
sourcedata=oridata;%[Data_training;Data_testing];
[nData,nData1]=size(sourcedata);
%[creditcardoutput,pendigits_Data]=modify_dataset_zero_class(sourcedata(:,end));
chunk_size=floor(nData/nFolds);
data=oridata;%[sourcedata(:,1:end-1) creditcardoutput];
A1=[];
B=[];
C=[];
D=[];
E=[];
mode='c'; %choose c for classification problem
F=[];
l=0;
Recall=[];
Precision=[];
buffer=[];
network=[];
prune_list_index=[];
counter=0;
ensemble=0;
prune_list=0;
covarianceinput=ones(ninput,noutput);
    del_list1=ones(1,ninput);
  traceinputweight=[];  
for  k=1:chunk_size:nData

lambdaD=min(1-exp(-counter/nFolds),0.001);
lambdaW=min(1-exp(-counter/(nFolds-1)),0.003);
confidenceinterval=lambdaD;
  if (k+chunk_size-1) > nData
        Data = data(k:nData,:);    %Tn = T(n:nTrainingData,:);
        %Block = size(Pn,1);             %%%% correct the block size
       % clear V;                        %%%% correct the first dimention of V 
    else
        Data = data(k:(k+chunk_size-1),:);   % Tn = T(n:(n+Block-1),:);
  end
[r,q]=size(Data);  
  [upperbound,upperboundlocation]=max(Data(:,1:ninput));
 [lowerbound,lowerboundlocation]=min(Data(:,1:ninput));

 
if ensemble==0
    %% Create the first hidden layer
paramet(1)=kprune; %conflict threshold
paramet(2)=kfs; %safety width
paramet(3)=vigilance; %vigilance parameter

fix_the_model=r;
[Weight,Center,Spread,rule,y,error,rules_significance,rules_novelty,datum_novelty,age,input_significance,population,time_index,born,time,classification_rate_testing,local_rate_of_change,feature_weights_evolution,population_class_cluster,count_samples,denumerator,forgetting_factor1,sigmapoints,focalpoints,feature_weights,cluster_novelty]=gClass_extended_multivariate(Data,fix_the_model,paramet,ninput,mode,drift,type_feature_weighting,input_selection,sample_deletion);
nRule=size(Center,1);
ensemble=ensemble+1;
network_parameters=nRule*noutput*(2*ninput+1)+nRule*(ninput)+nRule*(ninput)^(2);
network=struct('nRule',nRule,'Weight',Weight,'nParameters',network_parameters,'voting_weight',1,'ninput',ninput,'Center',Center,'Spread',Spread,'focalpoints',focalpoints,'sigmapoints',sigmapoints,'population',population,'population_class_clusters',population_class_cluster,'total_samples',count_samples,'decreasing_factor',initial);
for k3=1:noutput
covariance(1,:,k3)=0;
covariance(:,1,k3)=0;
end
covariance_old=covariance;
else
    counter=counter+1;
    covariance_old=covariance;

    storeoutput=zeros(size(Data,1),ensemble);
    storerawoutput=zeros(size(Data,1),noutput,ensemble);
    accuracy_matrix=zeros(size(Data,1),1);
    ensembleoutputtest=zeros(size(Data,1),1);
    %% Calculate Prequential Error
    totalrule=0;
    totalparameters=0;
    misclassification=0;
    classification=0;
      TP=0;
    FP=0;
    FN=0;
    for k1=1:size(Data,1)
       output=zeros(1,noutput);
           stream=Data(k1,:);
           stream(1:ninput)=del_list1.*stream(1:ninput);
  
       temporary_output=[];
    for m=1:ensemble
      if network(m).voting_weight~=0  
        if m==1          
    [ysem]=inference_ADL(stream,network(m).Weight,network(m).nRule,ninput,network(m).Center,network(m).Spread);
        else
         expanded_input=[stream(1:ninput) temporary_output stream(ninput+1:end)];

           [ysem]=inference_ADL(expanded_input,network(m).Weight,network(m).nRule,size(expanded_input,2)-noutput,network(m).Center,network(m).Spread);
        end
        for o=1:noutput
        if isnan(ysem(o))
        ysem(o)=0;
        end
        end
      else
          ysem=zeros(1,noutput);
      end

      [maxout,classlabel]=max(ysem);
      temporary_output=[temporary_output ysem];

         storeoutput(k1,m)=classlabel;

    if network(m).voting_weight~=0
        output(classlabel)=output(classlabel)+network(m).voting_weight;         
       

    end
                   
       for j=1:noutput
      storerawoutput(k1,j,m)=ysem(j);       
      end
    end
     [maxout1,trueclasslabel]=max(stream(ninput+1:end));
      [maxout,ensemblelabel]=max(output);
        ensembleoutputtest(k1)=ensemblelabel;
                if trueclasslabel==ensemblelabel
            classification=classification+1;
            if trueclasslabel==1
            TP=TP+1;
            end
        else
            accuracy_matrix(k1)=1;
            if trueclasslabel~=1
                FN=FN+1;
            else
                FP=FP+1;
            end
        end
    end
 A1(counter)=(classification)/size(Data,1);
  Precision(counter)=TP/(TP+FP);
Recall(counter)=TP/(TP+FN); 
 for m=1:ensemble
  totalrule=totalrule+network(m).nRule;
    totalparameters=totalparameters+network(m).nParameters;
 end
C(counter)=totalrule;
D(counter)=totalparameters;   
ensemblesize=0;
for i=1:ensemble
if network(i).voting_weight~=0
ensemblesize=ensemblesize+1;
end
end
E(counter)=ensemblesize;
%% Start the training process
tic
          %% Voting Weight Evaluation
                 for k1=1:size(Data,1)
               for m=1:ensemble
                   if network(m).voting_weight~=0
                   stream=Data(k1,:);
     [maxout1,trueclasslabel]=max(stream(ninput+1:end));
    if storeoutput(k1,m)==trueclasslabel
        %if network(m).decreasing_factor<1
        network(m).decreasing_factor=network(m).decreasing_factor+decreasingfactor;
        %end
        if network(m).decreasing_factor>1
            network(m).decreasing_factor=1;
        end
    network(m).voting_weight=min(network(m).voting_weight*(1+network(m).decreasing_factor),1);  
    else
        %   if network(m).decreasing_factor>0.0
        network(m).decreasing_factor=network(m).decreasing_factor-decreasingfactor;
         %  end
        if network(m).decreasing_factor<decreasingfactor
           network(m).decreasing_factor=decreasingfactor;
           end
        network(m).voting_weight=(network(m).voting_weight*network(m).decreasing_factor);
    end
                   end
               end
                 end
                 %% prune weight         
%            for i=1:ensemble
%            if network(i).voting_weight<0.001
%                network(i).voting_weight=0;
%            end
%            end
 %%Input Weighting Mechanism
 inputcovar=zeros(ninput,noutput);
     covarianceinput_old=covarianceinput;
for i=1:ninput
    for j=1:noutput
    temporary_IO=cov(Data(:,i),Data(:,ninput+j));
    inputcovar(i,j)=temporary_IO(1,2);
     covarianceinput(i,j)=(inputcovar(i,j));
     covarianceinputold=covarianceinput;
    end
end
            FCorrelation=zeros(ninput,noutput);
                       input_weight=zeros(1,ninput);
                       for i=1:ninput
                       for j=1:noutput
                                                  pearson=covarianceinput(i,j)/sqrt(var(Data(:,i))*var(Data(:,ninput+j)));
            FCorrelation(i,j)=(0.5*(var(Data(:,i))+var(Data(:,ninput+j)))-sqrt((var(Data(:,ninput+j))+var(Data(:,i)))^(2)-4*var(Data(:,i))*var(Data(:,ninput+j))*(1-pearson^(2))));
                       end
                       input_weight(i)=mean((FCorrelation(i,:)));
                       end
                       %
                       for i=1:ninput
                       if abs(input_weight(i))>threshold1
                           del_list1(i)=0;
                       end
                       end
                       %}
%                         [values,index]=sort(abs(input_weight),'descend');
%                          del_list1=ones(1,ninput);
%                         del_list1(index(subset+1:end))=0;
                       traceinputweight(counter,:)=del_list1;
                       Data(:,1:ninput)=del_list1.*Data(:,1:ninput);


%% ensemble merging mechanism
 outputcovar=zeros(ensemble,ensemble,noutput);
 for iter=1:ensemble
      
                        for iter1=1:ensemble
                            if network(iter).voting_weight~=0 && network(iter1).voting_weight~=0
                            for iter2=1:noutput
                            temporary=cov(storerawoutput(:,iter2,iter1),storerawoutput(:,iter2,iter));
                        outputcovar(iter,iter1,iter2)=temporary(1,2);
                        covariance(iter,iter1,iter2)=(covariance_old(iter,iter1,iter2)*(counter-1)+(((counter-1)/counter)*outputcovar(iter,iter1,iter2)))/counter;
                            end
                            end
                        end
 end
 %
 if (ensemblesize)>1 && merging==1
                        merged_list=[];
                      
                for l=0:ensemble-2
        for hh=1:ensemble-l-1
            if network(end-l).voting_weight~=0 || network(hh).voting_weight~=0
            MCI=zeros(1,noutput);
            for o=1:noutput
            pearson=covariance(end-l,hh,o)/sqrt(covariance(end-l,end-l,o)*covariance(hh,hh,o));
            MCI(o)=(0.5*(covariance(hh,hh,o)+covariance(end-l,end-l,o))-sqrt((covariance(hh,hh,o)+covariance(end-l,end-l,o))^(2)-4*covariance(end-l,end-l,o)*covariance(hh,hh,o)*(1-pearson^(2))));
            end
       
                                           if max(abs(MCI))<threshold %&& counter-network(end-l).born>5 && counter-network(hh).born>5 %(max(MCI)<0.1 & max(MCI)>0) & (max(MCI)>-0.1 & max(MCI)<0)
           if isempty(merged_list)
          merged_list(1,1)=ensemble-l;
          merged_list(1,2)=hh;
          else
            No=find(merged_list(:,1:end-1)==ensemble-l);
            No1=find(merged_list(:,1:end-1)==hh);
            if isempty(No) && isempty(No1)
          merged_list(end+1,1)=ensemble-l;
          merged_list(end+1,2)=hh;
            end
           end
           break
                               end 
        end
        end
                end
                del_list=[];
                                    for i=1:size(merged_list,1)
                    No2=find(merged_list(i,:)==0);
                    if isempty(No2)
                                            if network(merged_list(i,1)).voting_weight>network(merged_list(i,2)).voting_weight
                      a=merged_list(i,1);
                      b=merged_list(i,2);
                      else
                        b=merged_list(i,1);
                      a=merged_list(i,2);    
                                            end
                                            del_list=[del_list b]; 
                    end
                                    end
                    if isempty(del_list)==false && network(del_list).voting_weight~=0
                    network(del_list).voting_weight=0;
                    end
                    prune_list=prune_list+length(del_list);
                    prune_list_index=[prune_list_index del_list];
 end
 %}
%% Drift Detection
        Subject=Data(:,1:ninput);
        a=find(accuracy_matrix);
%         if isempty(a)==false
%         Zstat=1/length(find(accuracy_matrix));%mean(Subject);%length(find(accuracy_matrix))/length(accuracy_matrix);
%         else
%             Zstat=0;
%         end
Zstat=length(find(accuracy_matrix))/length(accuracy_matrix);
    cuttingpoint=0;
                [Zupper,Zupper2]=max(accuracy_matrix);
        [Zlower,Zlower2]=min(accuracy_matrix);

        for cut=1:size(Data,1)
%         b=find(accuracy_matrix(1:cut));
%         if isempty(b)==false
%         Xstat=1/length(find(accuracy_matrix(1:cut)));%mean(Subject(1:cut,:));%length(find(accuracy_matrix(1:cut)))/cut;
%         else
%             Xstat=0;
%         end
Xstat=length(find(accuracy_matrix(1:cut)))/cut;
        [Xupper,Xupper1]=max(accuracy_matrix(1:cut));
        [Xlower,Xlower1]=min(accuracy_matrix(1:cut));
        Xbound=(Xupper-Xlower)*sqrt(((r)/(2*cut*(r))*reallog(1/confidenceinterval)));
        Zbound=(Zupper-Zlower).*sqrt(((r)/(2*cut*(r))*reallog(1/confidenceinterval)));
        if mean(Xbound+Xstat)>=mean(Zstat+Zbound) 
            cuttingpoint=cut;
%           c=find(accuracy_matrix(cuttingpoint+1:end));
%           if isempty(c)==false
%               Ystat=1/length(find(accuracy_matrix(cuttingpoint+1:end)));%mean(Subject(cuttingpoint+1:end,:));%length(find(accuracy_matrix(cuttingpoint+1:end)))/r-cuttingpoint;
%           else
%               Ystat=0;
%           end
Ystat=length(find(accuracy_matrix(cuttingpoint+1:end)))/(r-cuttingpoint);
                      [Yupper,Yupper1]=max(accuracy_matrix(cuttingpoint+1:end));
        [Ylower,Ylower1]=min(accuracy_matrix(cuttingpoint+1:end));
         Ybound=(Zupper-Zlower).*sqrt(((r-cuttingpoint)/(2*cuttingpoint*(r-cuttingpoint)))*reallog(1/lambdaD));
          Ybound1=(Zupper-Zlower).*sqrt(((r-cuttingpoint)/(2*cuttingpoint*(r-cuttingpoint)))*reallog(1/lambdaW));
            break
       
        end
        end
% signal=0;
% if Yupper==Ylower && Yupper==0
% signal=1;
% end
expandedfactor=[];
Data_fix=[];
  for m=1:ensemble
            expandedfactor=[expandedfactor storerawoutput(:,:,m)];
  end
  nodrift=1;
  if cuttingpoint~=0
if (mean(abs(Xstat-Ystat)))>mean(abs(Ybound)) && cuttingpoint>1 && cuttingpoint<r
    %%introduce a new layer
    nodrift=0;
    if isempty(buffer)
Data_fix=[Data(:,1:ninput) expandedfactor Data(:,ninput+1:end)];
 fix_the_model=size(Data,1);
    else
        
    updatebuffer=[buffer(:,1:ninput) zeros(size(buffer,1),size(expandedfactor,2)) buffer(:,ninput+1:end)];
   Data_fix=[Data(:,1:ninput) expandedfactor Data(:,ninput+1:end);updatebuffer]; 
    fix_the_model=size(Data,1)+size(buffer,1);
    end
    ndimension=size(Data_fix,2)-noutput;
paramet(1)=kprune; %conflict threshold
paramet(2)=kfs; %safety width
paramet(3)=vigilance; %vigilance parameter
[Weight,Center,Spread,rule,y,error,rules_significance,rules_novelty,datum_novelty,age,input_significance,population,time_index,born,time,classification_rate_testing,local_rate_of_change,feature_weights_evolution,population_class_cluster,count_samples,denumerator,forgetting_factor1,sigmapoints,focalpoints,feature_weights,cluster_novelty]=gClass_extended_multivariate(Data_fix,fix_the_model,paramet,ndimension,mode,drift,type_feature_weighting,input_selection,sample_deletion);
nRule=size(Center,1);
buffer=[];
ensemble=ensemble+1;
nInput=ndimension;
network_parameters=nRule*noutput*((2*nInput)+1)+nRule*(nInput)+nRule*(nInput)^(2);
network=[network;struct('nRule',nRule,'Weight',Weight,'nParameters',network_parameters,'voting_weight',1,'ninput',nInput,'Center',Center,'Spread',Spread,'focalpoints',focalpoints,'sigmapoints',sigmapoints,'population',population,'population_class_clusters',population_class_cluster,'total_samples',count_samples,'decreasing_factor',initial);];
for k3=1:noutput
covariance(:,ensemble,k3)=0;
covariance(ensemble,:,k3)=0;
end
elseif (mean(abs(Xstat-Ystat)))>mean(abs(Ybound1)) && (mean(abs(Xstat-Ystat)))<mean(abs(Ybound)) && cuttingpoint>1 && cuttingpoint<r
    %%accummulate data buffer
    buffer=[buffer;Data];
    nodrift=0;
else
    %%update the winning rule
    nodrift=1;
  
end
  end
if cuttingpoint==0 || nodrift==1
    transfer=zeros(1,ensemble);
    for i=1:ensemble
    transfer(i)=network(i).voting_weight;
    end
      [winning,winner]=max(transfer);
paramet(1)=kprune; %conflict threshold
paramet(2)=kfs; %safety width
paramet(3)=vigilance; %vigilance parameter
buffer=[];
if winner~=1
Data_fix=[Data(:,1:ninput) expandedfactor(:,1:noutput*(winner-1)) Data(:,ninput+1:end)];
else
    Data_fix=Data;
end
fix_the_model=r;
ndimension=size(Data_fix,2)-noutput;
[Weight,Center,Spread,rule,y,error,rules_significance,rules_novelty,datum_novelty,age,input_significance,population,time_index,born,time,classification_rate_testing,local_rate_of_change,feature_weights_evolution,population_class_cluster,count_samples,denumerator,forgetting_factor1,sigmapoints,focalpoints,feature_weights,cluster_novelty]=gClass_extended_multivariate_update(Data_fix,fix_the_model,paramet,ndimension,mode,drift,type_feature_weighting,input_selection,sample_deletion,network(winner));

nRule=size(Center,1);
if winner==1
nInput=ninput;
else
    nInput=ndimension;
end
network_parameters=nRule*noutput*((2*nInput)+1)+nRule*(nInput)+nRule*(nInput)^(2);
replacement=struct('nRule',nRule,'Weight',Weight,'nParameters',network_parameters,'voting_weight',1,'ninput',nInput,'Center',Center,'Spread',Spread,'focalpoints',focalpoints,'sigmapoints',sigmapoints,'population',population,'population_class_clusters',population_class_cluster,'total_samples',count_samples+network(winner).total_samples,'decreasing_factor',initial);
network(winner)=replacement;
end
time=toc;
H(counter)=time;
F(counter)=count_samples;
end
end
Brat=mean(A1);
Bdev=std(A1);
Crat=mean(C);
Cdev=std(C);
Drat=mean(D);
Ddev=std(D);
%
Erat=mean(E);
Edev=std(E);
Hrat=mean(H);
Hdev=std(H);