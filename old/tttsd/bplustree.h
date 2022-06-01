#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
//L_ctermid是一个和系统调用char * ctermid(char *)相关的宏，该值为一个int，其大小足够容纳ctermid()返回的字符串，
//也就是返回值char * 的长度不会超过L_ctermid，但是NOPE是没有的意思，所以这里可能表示没有定义L_ctermid.
#ifdef L_ctermidNOPE
#include <unistd.h>
#include <fcntl.h>
#define LOCK //定义了锁，增加了并发的操作可能
#endif

typedef unsigned long long uint64_t;
typedef unsigned int uint32_t;
typedef unsigned char uint8_t;
typedef struct db db;

#define SIZEOF_LONG sizeof(uint64_t)                         //表示一个地址占用的字节数（这里的地址就是在文件中的位置）
#define _HASH 4                                              //一个node里面，每个项item所占用的字节数，也就是key所占用的字节数
#define _ORDER 99                                            //阶数，可以理解为叉数
#define _WIDTH 1 + _HASH *_ORDER + SIZEOF_LONG *(_ORDER + 1) //每个node所占用的空间大小，1个字节为0/1表示是否是叶子节点，_ORDER个项占_ORDER*_HASH字节，（_ORDER+1）个指针占（_ORDER+1）*8；（地址数量比项数量多1，每个地址用8字节表示）
#define _DEPTH 10                                            //树的深度
#define _MAX 0xf4240                                         //1+e6,一百万（一个value的上限）

struct db //在fp表示的存储B+树的文件中，从根节点出发到达某个节点的路径上的节点和各节点的地址（在文件中的位置）
{
  FILE *fp;
  unsigned char path[_DEPTH][_WIDTH];
  uint64_t node_addrs[_DEPTH];
#ifdef LOCK
  struct flock fl;
#endif
};

// void to_big(unsigned char *, uint64_t);
// uint64_t from_big(unsigned char *);
// void node_split(db *, int, unsigned char);
// void _insert(db *, unsigned char *, int, uint64_t, uint64_t, int);
// void put(db *, unsigned char *, unsigned char *);
// uint64_t search(db *, unsigned char *, int *);
// void db_init(db *, const char *);

#ifdef LOCK
int db_lock(db *);
int db_unlock(db *);
#endif


#ifdef LOCK
int db_lock(db *db)
{
  db->fl.l_type = F_WRLCK;
  db->fl.l_whence = SEEK_SET;
  db->fl.l_start = 0;
  db->fl.l_len = 0;
  db->fl.l_pid = getpid();
  return fcntl((db->fp)->_file, F_SETLKW, &(db->fl)); //给文件设置写锁
}

int db_unlock(db *db)
{
  db->fl.l_type = F_UNLCK;
  fcntl((db->fp)->_file, F_SETLK, &(db->fl)); //释放加在文件上的锁（读锁或者写锁）
}
#endif

/**
 * 将64位整数转换成大端表示形式，并放到数组中。这种方式是便于人理解的，高位在前，低位在后，
 * 便于统一比较和计算大小（主要用于key的比较，从前往后遍历即可确定谁大谁小）。
 * 例如：uint64_t val =1234567890在内存或者文件上的二进制表示为：d2,02,96,49,00,00,00,00；
 * 这是小端表示方式，也就是低位在前，高位在后，无法直接用于比较整数大小；如果换成大端表示，则为：
 * 00,00,00,00,49,96,02,d2，而用16进制表示1234567890这个数，也是0x499602d2.
 */
void to_big(unsigned char *buf, uint64_t val);
/**
 * 将使用大端形式来表示整数的一个无符号字符数组拼装成一个64位整数。
 */
uint64_t
from_big(unsigned char *buf);
/**
 * 处理节点分裂的情况；分成两种分裂：叶子节点分裂和内部节点分裂。注意：node_split()和_insert()是相互调用的，这是这套代码的精妙之处。
 */
void node_split(db *db, int index, unsigned char isleaf);
/**
 * 根据db的描述，将key和记录对应内容的地址addr构成的项插入到index层，同时附带右子树根节点地址，以及该项要插入的node是否为叶子节点。
 * 这里的难点是_insert()和node_split()函数相互调用。理解起来比较麻烦。
 */
void _insert(db *db, unsigned char *key, int index, uint64_t addr, uint64_t rptr, int isleaf);
/**
 * 查找key所在的位置（已经存在），或者应该所在的位置（待插入的位置），并将位置信息保存在r_index中，表示从根节点到该位置的路径。
 * @param key 待查找的key（使用字符串是一种通用表示，前面的to_big也是将整数标准化成可比较的字符串）
 * @param r_index [out] 从根节点到key的路径
 */
uint64_t
search(db *db, unsigned char *key, int *r_index);
/**
 * 在B+树中插入一项key和value.并将相关信息保留在db中。
 * 我猜测文件的构成应该是开头8个字节表示root节点所在的位置，后面跟着一系列节点和记录（value）。
 * 每个记录包含1个字节的有效位，key_len和key构成的key字段，value_len和value构成的value字段.
 * B+树在使用的时候是从文件中读取出来（根据root节点及其左右子树地址），然后在内存中构建的，B+树本身也存到文件中了。
 */
void putkeyvalue(db *db, unsigned char *key, unsigned char *value);

char* getvalue(db *db, unsigned char *key);

/**
*delete并不真实删除数据，只是把记录的第一个字节从1变成了0，表示数据无效。
*/
void deletedb(db *db, unsigned char *key);

void db_init(db *db, const char *name);

void db_close(db *db);
unsigned char *random_str();
#ifdef __cplusplus
}
#endif