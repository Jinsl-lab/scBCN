BatchKL=function(df,dimensionData=NULL,replicates=200,n_neighbors=100,n_cells=100,batch="BatchID"){
  set.seed(1)
  if (is.null(dimensionData)){
        tsnedata=as.matrix(df[,c("tSNE_1","tSNE_2")])
  }else{
        tsnedata=as.matrix(dimensionData)
  }
  batchdata=factor(as.vector(df[,batch]))
  table.batchdata=as.matrix(table(batchdata))[,1]
  tmp00=table.batchdata/sum(table.batchdata)
  n=dim(df)[1]
  KL=sapply(1:replicates,function(x){
    bootsamples=sample(1:n,n_cells)
    nearest=nabor::knn(tsnedata,tsnedata[bootsamples,],k=min(5*length(tmp00),n_neighbors))
    KL_x=sapply(1:length(bootsamples),function(y){
      id=nearest$nn.idx[y,]
      tmp=as.matrix(table(batchdata[id]))[,1]
      tmp=tmp/sum(tmp)
      return(sum(tmp*log2(tmp/tmp00),na.rm = T))
    })
    return(mean(KL_x,na.rm = T))
  })
  return(mean(KL))
}