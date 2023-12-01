library("xlsx")
library("survival")

cal.age.enrichment <- function(age, cluster){
  age.test.result = kruskal.test(as.numeric(age_file[,2]), as.numeric(cluster_file[,2]))
  return(age.test.result)
}

cal.discrete.enrichment <- function(file_dir){
  para_tb = table(as.data.frame(file_dir[,2:3]))
  print(para_tb)
  enrichment.result <- chisq.test(para_tb)
  return(enrichment.result)
} #离散值  卡方检验

#设置路径
file_dir = "C:/Users/79354/Desktop/M2CGCN/"
param_dir = paste0(file_dir, "_param_comb")
enrich_dir = paste0(file_dir)
enrichdata_dir="C:/Users/79354/Documents/test/BIC/BREAST"


#读聚类结果的标签
label=read.xlsx("C:/Users/79354/Desktop/M2CGCN/test/BIC4.xlsx",sheetIndex = 1,row.names = 1)



# 生存分析 --------------------------------------------------------------------

#读Survival生存数据
surv_name = paste0(enrichdata_dir,"_Survival.txt")    #paste0默认分隔符sep=""
print(surv_name)

survdata<-read.table(file = surv_name, header = T,fill = TRUE)   #fill=TRUE情况下，行有长度不等的空白领域隐式添加。 
survdata$labels=label[,1]  #添加label列，将数据合并
survdata$Survival=as.numeric(survdata$Survival)
survdata$Death=as.numeric(survdata$Death)
survdata$labels=as.numeric(survdata$labels)  #将生存时间、死亡状态和标签都转变为数值型

survresult <- survdiff(Surv(Survival,Death)~labels, data=survdata)
p.val <- 1 - pchisq(survresult$chisq, length(survresult$n) - 1)  #自由度为（聚类标签数-1）的卡方统计量为34.3 
print(survresult)
print(p.val)#生存分析的显著性

p.val=as.numeric(p.val)
computep=-log10(p.val)    #取负对数


# 富集分析 --------------------------------------------------------------------

enrich_num=0   #数量初始化为0
age = paste0(enrichdata_dir,"_Age.xlsx")
age_file <- read.xlsx(age, sheetIndex = 1)

cluster_name = "C:/Users/79354/Desktop/M2CGCN/test/BIC4.xlsx"
cluster_file <- read.xlsx(cluster_name, sheetIndex = 1)

age_test_res = cal.age.enrichment(age_file, cluster_file)
print(age_test_res)
p_value = age_test_res$p.value
if(p_value<0.0083)   #0.05÷6
  enrich_num=enrich_num+1

print("_______________________________________________________________________________")

gender_name = paste0(enrichdata_dir,"_gender.xlsx")
gender_file = read.xlsx(gender_name,sheetIndex = 1)
clustering = cluster_file[,2]

gender_file$labels=clustering  #添加labels列，将需要数据合并
gender.pval = cal.discrete.enrichment(gender_file)
print(gender.pval)
p_val = as.matrix(gender.pval)
p_value = as.data.frame(p_val[3,])
if(p_value<0.0083)
  enrich_num=enrich_num+1

print("_______________________________________________________________________________")

path_M_name = paste0(enrichdata_dir,"_pathM.xlsx")
print(path_M_name)

pathM_file = read.xlsx(path_M_name, sheetIndex = 1)
pathM_file$labels=clustering
pathM.pval = cal.discrete.enrichment(pathM_file)
print(pathM.pval)
p_val = as.matrix(pathM.pval)
p_value = as.data.frame(p_val[3,])
if(p_value<0.0083)
  enrich_num=enrich_num+1

print("_______________________________________________________________________________")

path_N_name = paste0(enrichdata_dir,"_pathN.xlsx")
print(path_N_name)

pathN_file = read.xlsx(path_N_name, sheetIndex = 1)
pathN_file$labels=clustering
pathN.pval = cal.discrete.enrichment(pathN_file)
print(pathN.pval)
p_val = as.matrix(pathN.pval)
p_value = as.data.frame(p_val[3,])
if(p_value<0.0083)
  enrich_num=enrich_num+1

print("_______________________________________________________________________________")

path_T_name = paste0(enrichdata_dir,"_pathT.xlsx")
print(path_T_name)

pathT_file = read.xlsx(path_T_name, sheetIndex = 1)
pathT_file$labels=clustering
pathT.pval = cal.discrete.enrichment(pathT_file)
print(pathT.pval)
p_val = as.matrix(pathT.pval)
p_value = as.data.frame(p_val[3,])
if(p_value<0.0083)
  enrich_num=enrich_num+1

print("_______________________________________________________________________________")

path_stage_name = paste0(enrichdata_dir,"_path_stage.xlsx")
print(path_stage_name)

path_stage_file = read.xlsx(path_stage_name, sheetIndex = 1)
path_stage_file$labels=clustering
path_stage.pval = cal.discrete.enrichment(path_stage_file)
print(path_stage.pval)
p_val = as.matrix(path_stage.pval)
p_value = as.data.frame(p_val[3,])
if(p_value<0.0083)
  enrich_num=enrich_num+1
write.table(computep, paste0(enrich_dir,"surv_pval", ".txt"), row.names = F, col.names = F,append=TRUE)
write.table(enrich_num, paste0(enrich_dir,"enrich_num", ".txt"), row.names = F, col.names = F,append=TRUE)
