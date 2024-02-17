#include <stdio.h>
#include "stdbool.h"
/*
 结构体就是多种数据类型的几何体，形成了新的类型

 */
typedef enum {
    Male,
    Female,
}SexType;


struct student{
    char name[30];/* 姓名 */
    int  age; /* 年龄 */
    bool sex ; /* 性别 0：男性 ，1 ：女性*/
};

int main() {
    /* 结构体的定义方式 */
    struct student s1 = {
            .name = "zhang san ",
            .age = 18,
            .sex = Male,
    };

    struct student s2 = {"lisi",20,Female};

    struct student s3 = {
        .name = "wangwu",
    };
    s3.age = 20;
    s3.sex = Female;

    printf("name:%s age:%d sex:%s \n",s1.name,s1.age,s1.sex == Male ?"Male" : "Female");
    printf("name:%s age:%d sex:%s \n",s2.name,s2.age,s2.sex == Male ?"Male" : "Female");
    printf("name:%s age:%d sex:%s \n",s3.name,s3.age,s3.sex == Male ?"Male" : "Female");
    return 0;
}
