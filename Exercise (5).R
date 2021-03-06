getwd()
setwd("F:/MICA/Term 4/AMMA/Data/data_2017/data_2017")
d1=read.table("student-mat.csv",sep=";",header=TRUE)
d2=read.table("student-por.csv",sep=";",header=TRUE)

d3=merge(d1,d2,by=c("school","sex","age","address","famsize","Pstatus","Medu","Fedu","Mjob","Fjob","reason","nursery","internet"))
print(nrow(d3)) # 382 students
View(d1)
View(d2)
View(d3)

##### Part A #########
female_mean=0
male_mean=0

for(i in 1:length(d3$G3.x))
{
  if(d3$sex[i]=="F")
  {  female_mean = female_mean + d3$G3.x[i] + d3$G3.y[i]
  }
  else
  {
    male_mean = male_mean + d3$G3.x[i] + d3$G3.y[i]
  }
}
male_mean = male_mean/length(d3$G3.x)
female_mean = female_mean/length(d3$G3.x)

male_mean
female_mean  
####### Part A End #########
####### Part B #########
max_G1 = max(d3$G1.x + d3$G1.y)
max_G2 = max(d3$G2.x + d3$G2.y)
max_G3 = max(d3$G3.x + d3$G3.y)

for(j in 1:length(d3$G3.x))
{  
  if(d3$G1.x[j]+d3$G1.y[j] == max_G1)
  {Guardian_G1 = d3$guardian.x[j]
  Gender_G1 = d3$sex[j]
  }
}  
Guardian_G1
Gender_G1

for(j in 1:length(d3$G3.x))
{  
  if(d3$G2.x[j]+d3$G2.y[j] == max_G1)
  {Guardian_G2 = d3$guardian.x[j]
  Gender_G2 = d3$sex[j]
  }
}  
Guardian_G2
Gender_G2

for(j in 1:length(d3$G3.x))
{  
  if(d3$G3.x[j]+d3$G3.y[j] == max_G1)
  {Guardian_G3 = d3$guardian.x[j]
  Gender_G3 = d3$sex[j]
  }
}  
Guardian_G3
Gender_G3
print("Combo 1")
print(Guardian_G1)
print(Gender_G1)

print("Combo 2")
print(Guardian_G2)
print(Gender_G2)

print("Combo 3")
print(Guardian_G3)
print(Gender_G3)

###### Q2 ##############
#####Dividing Data#####
library(dplyr)
d3$absences<-d3$absences.x+d3$absences.y
d3$quartile <- ntile(d3$absences, 4)  
d3$quartile<-as.factor(d3$quartile)
summary(d3$quartile)

##### Finding Grades ######
attendance.morethan75_grades=0
attendance.between50and75_grades=0
attendance.between25and50_grades=0
attendance.lessthan25_grades=0

for(i in 1:length(d3$G3.x))
{
  if(d3$quartile[i]=="1")
  {  attendance.morethan75_grades= attendance.morethan75_grades + d3$G3.x[i] + d3$G3.y[i]
  
 } else
  if(d3$quartile[i]=="2")
  {attendance.between50and75_grades= attendance.between50and75_grades + d3$G3.x[i] + d3$G3.y[i]
  
  }else
    if(d3$quartile[i]=="3")
      {attendance.between25and50_grades= attendance.between25and50_grades + d3$G3.x[i] + d3$G3.y[i]
    
 } else
    if(d3$quartile[i]=="4")
    {attendance.lessthan25_grades= attendance.lessthan25_grades + d3$G3.x[i] + d3$G3.y[i]}
}
summary(d3$quartile)
mean_morethan75<-attendance.morethan75_grades/96
mean_between50and75<-attendance.between50and75_grades/95
mean_between25and50<-attendance.between25and50_grades/96
mean_lessthan25<-attendance.lessthan25_grades/95

print("Average Grade for attendance greater than 75")
mean_morethan75
print("Average Grade for attendance between 50 and 75")
mean_between50and75
print("Average Grade for attendance between 25 and 50")
mean_between25and50
print("Average Grade for attendance less than 25")
mean_lessthan25

#############################################################################
      
      
      