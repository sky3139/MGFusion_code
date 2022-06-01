#include "bplustree.h"
/*
 * To conver to an append-only design, only the 'fwrite' in the '_insert' function
 * must be changed. All other writes (save for delete), are appended to the end.
 * Also, the root address must instead be written to and searched for at the end.
 * 
一个完整的B+树代码及详细注释
 */

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

#define SIZEOF_LONG sizeof(uint64_t)                         //表示一个地址占用的字节数（这里的地址就是在文件中的位置）
#define _HASH 4                                              //一个node里面，每个项item所占用的字节数，也就是key所占用的字节数
#define _ORDER 99                                            //阶数，可以理解为叉数
#define _WIDTH 1 + _HASH *_ORDER + SIZEOF_LONG *(_ORDER + 1) //每个node所占用的空间大小，1个字节为0/1表示是否是叶子节点，_ORDER个项占_ORDER*_HASH字节，（_ORDER+1）个指针占（_ORDER+1）*8；（地址数量比项数量多1，每个地址用8字节表示）
#define _DEPTH 10                                            //树的深度
#define _MAX 0xf4240                                         //1+e6,一百万（一个value的上限）

// struct db //在fp表示的存储B+树的文件中，从根节点出发到达某个节点的路径上的节点和各节点的地址（在文件中的位置）
// {
//   FILE *fp;
//   unsigned char path[_DEPTH][_WIDTH];
//   uint64_t node_addrs[_DEPTH];
// #ifdef LOCK
//   struct flock fl;
// #endif
// };

// void to_big(unsigned char *, uint64_t);
// uint64_t from_big(unsigned char *);
// void node_split(db *, int, unsigned char);
// void _insert(db *, unsigned char *, int, uint64_t, uint64_t, int);
// void put(db *, unsigned char *, unsigned char *);
// uint64_t search(db *, unsigned char *, int *);
//  char *getvalue(db *, unsigned char *);
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
void to_big(unsigned char *buf, uint64_t val)
{
  int i=0;
  for (i = 0; i < sizeof(uint64_t); ++i)
  {
    buf[i] = (val >> (56 - (i * 8))) & 0xff;
  }
}

/**
 * 将使用大端形式来表示整数的一个无符号字符数组拼装成一个64位整数。
 */
uint64_t
from_big(unsigned char *buf)
{
  uint64_t val = 0;
  int i;
  for (i = 0; i < sizeof(uint64_t); ++i)
    val |= (uint64_t)buf[i] << (56 - (i * 8));
  return val;
}

/**
 * 处理节点分裂的情况；分成两种分裂：叶子节点分裂和内部节点分裂。注意：node_split()和_insert()是相互调用的，这是这套代码的精妙之处。
 */
void node_split(db *db, int index, unsigned char isleaf)
{
  unsigned char *node = db->path[index];     //待分裂的节点
  unsigned char *lnode = malloc(_WIDTH + 1); //左子树的根节点（或者说左子节点）
  unsigned char *rnode = malloc(_WIDTH + 1); //右子树的根节点（或者说右子节点）
  memset(lnode, 0, _WIDTH + 1);
  memset(rnode, 0, _WIDTH + 1);

  int split = (_HASH + SIZEOF_LONG) * (_ORDER >> 1) + 1; //计算分裂点的位置
  memcpy(lnode, node, split);                            //将node中位于分裂点左侧的数据复制到lnode中，充当左子树的根节点
  rnode[0] = isleaf;                                     //右子树的根节点的类型与待分裂节点一致。如果待分裂的节点是叶子节点，也由于分裂是向上生长，留下来的节点肯定还是叶子节点；
  //可以推测，左子树根节点的类型与右子树的根节点类型是一致的，这里只是没有单独给lnode[0]赋值。
  isleaf ? memcpy(rnode + 1, node + split, _WIDTH - split) : memcpy(rnode + 1, node + split + SIZEOF_LONG + _HASH, _WIDTH - (split + SIZEOF_LONG + _HASH));
  //如果是叶子节点，则中间的项会保留在右子树中，而不会分裂到上一层，只是把key复制到上层而已，本项仍保留在右子树中；如果是内部节点，则中间项是要提升到上层的。

  fseeko(db->fp, 0, SEEK_END);
  uint64_t addr = ftello(db->fp);
  fwrite(rnode, 1, _WIDTH, db->fp); //把右子树的根节点写入文件
                                    //叶子节点保留平级右指针，内部节点应该不用右指针
  //to_big(lnode + split, addr);//此代码错误，会在平级内部节点之间用指针链起来，造成部分内部节点和叶子节点没有父节点，造成无法检索。
  isleaf ? to_big(lnode + split, addr) : memcpy(lnode, node, split + SIZEOF_LONG); //将右子树根节点地址写入到左子树的最后一个有效位置，相当于把它们串起来了。不是只把叶子那层串起来吗？这里怎么把内部节点层也串起来了？
  //要升上去的项肯定是是放到内部节点中的，所以这里把待升上去的key抽出来，给其配对的左右指针分别指向lnode的地址和rnode的地址。

  // to_big(lnode + split, addr);//将右子树根节点地址写入到左子树的最后一个有效位置，相当于把它们串起来了。不是只把叶子那层串起来吗？这里怎么把内部节点层也串起来了？
  //要升上去的项肯定是是放到内部节点中的，所以这里先把待升上去的项抽出来，主要是key和指向value的地址啦
  unsigned char *key = malloc(_HASH + 1);
  memset(key, 0, _HASH + 1);
  memcpy(key, node + split + SIZEOF_LONG, _HASH);
  memcpy(db->path[index], lnode, _WIDTH); //db->path[index]用左子树根节点代替
  if (index > 0)
  {
    _insert(db, key, index - 1, db->node_addrs[index], addr, 0); //把key和记录对应内容的地址db->node_addrs[index]构成的项，添加到上一层的node中，这一项对应的节点为内部节点，该项的右子树地址为addr
  }
  else //index==0说明是第一次分裂，也就是最早的根节点填充满了，需要分裂，由最初的一层变成两层。由一个root节点变成3个节点，一个root和两个子节点
  {
    unsigned char *root = malloc(_WIDTH + 1); //新生成的root节点
    memset(root, 0, _WIDTH + 1);
    root[0] = 0;
    to_big(root + 1, db->node_addrs[0]);          //左子节点的地址
    strncpy(root + 1 + SIZEOF_LONG, key, _HASH);  //key
    to_big(root + 1 + SIZEOF_LONG + _HASH, addr); //右子节点的地址
    fseeko(db->fp, 0, SEEK_END);
    addr = ftello(db->fp);
    fwrite(root, 1, _WIDTH, db->fp); //将新生成的root写入到文件末尾
    fseeko(db->fp, 0, SEEK_SET);
    fwrite(&addr, 1, SIZEOF_LONG, db->fp); //更新文件中记录的整个树的根节点，用新的root节点的地址覆盖原来记录的root的地址
  }
}

/**
 * 根据db的描述，将key和记录对应内容的地址addr构成的项插入到index层，同时附带右子树根节点地址，以及该项要插入的node是否为叶子节点。
 * 这里的难点是_insert()和node_split()函数相互调用。理解起来比较麻烦。
 */
void _insert(db *db, unsigned char *key, int index, uint64_t addr, uint64_t rptr, int isleaf)
{
  if (_HASH > strlen(key))
  {
    unsigned char *okey = key;
    key = malloc(_HASH + 1);
    memset(key, 0x61, _HASH + 1);
    strncpy(key, okey, strlen(okey));
  }
  unsigned char *node = db->path[index]; //先读出插入位置所在的node
  int i = SIZEOF_LONG + 1;
  for (; i < _WIDTH; i += (_HASH + SIZEOF_LONG)) //从node中找到key的具体位置
  {
    if (node[i] == 0) //该key没有找到；node[i]为null表示node中的最后一项，可以插入在该位置，更新key/value即可
    {
      i -= SIZEOF_LONG;
      to_big(node + i, addr); //把addr写到key前面的8个字节中
      i += SIZEOF_LONG;
      strncpy(node + i, key, _HASH); //把key写进去
      if (!isleaf)                   //不是叶子节点，说明是内部节点，把该项对应的右子树的根接单写入到node中。
      {
        i += _HASH;
        to_big(node + i, rptr);
      }
      break;
    }
    if (!strncmp(node + i, key, _HASH)) //key在当前node中已经存在了
    {
      if (isleaf) //如果是叶子节点，则更新前面的value的地址，相当于覆盖以前的值。
      {
        i -= SIZEOF_LONG;
        to_big(node + i, addr);
      }
      break;
    }

    if (strncmp(node + i, key, _HASH) > 0) //node[i]>key,说明要往前插入；
    {
      unsigned char *nnode = malloc(_WIDTH + 1); //重新申请一个node
      memset(nnode, 0, _WIDTH + 1);
      i -= SIZEOF_LONG;
      int j;
      for (j = 0; j < i; ++j)
      {
        nnode[j] = node[j];
      }
      to_big(nnode + i, addr);
      i += SIZEOF_LONG;
      strncpy(nnode + i, key, _HASH);
      i += _HASH;
      if (!isleaf) //如果不是叶子节点，则需要把右子树的根地址插入
      {
        to_big(nnode + i, rptr);
        i += SIZEOF_LONG;
        j += SIZEOF_LONG;
      }

      for (; i < _WIDTH; ++i, ++j)
      {
        nnode[i] = node[j];
      }
      memcpy(db->path[index], nnode, _WIDTH);
      break;
    }
  }

  if (node[(_WIDTH - (_HASH + SIZEOF_LONG))] != 0) //前面插入一项完毕后，要判断当前node是否已经满了（最后一项的key是否为0），是否需要分裂
  {
    isleaf ? node_split(db, index, 1) : node_split(db, index, 0);
  }
  fseeko(db->fp, db->node_addrs[index], SEEK_SET);
  fwrite(db->path[index], 1, _WIDTH, db->fp); //如果只是更新（或者插入后没有引起分裂），则需要把更新的node写入到文件中进行覆盖。
}

/**
 * 查找key所在的位置（已经存在），或者应该所在的位置（待插入的位置），并将位置信息保存在r_index中，表示从根节点到该位置的路径。
 * @param key 待查找的key（使用字符串是一种通用表示，前面的to_big也是将整数标准化成可比较的字符串）
 * @param r_index [out] 从根节点到key的路径
 */
uint64_t
search(db *db, unsigned char *key, int *r_index)
{
  if (_HASH > strlen(key))
  { //如果key的长度不够，则重建key，后面字符a，这个我不赞同，如果仅仅用于纯小写英文字母组成的key判断，是没问题的，但是如果包含了大写字母或者数字以及其他字符，这个就会出错。
    unsigned char *okey = key;
    key = malloc(_HASH + 1);
    memset(key, 0x61, _HASH + 1);
    strncpy(key, okey, strlen(okey));
  }
  uint64_t r_addr; //指向root节点的指针

  int i = SIZEOF_LONG + 1; //node内部的游标，从0到_WIDTH，每次移动到key的开始位置或者存放左子树/右子树指针的地址。
  unsigned char isleaf;    //是否是叶子节点，通过node中的第一个字节是0或者1来判断。
  int index = 0;           //node内部的index计数，i是字节级别，index是单算key为第几个的，从0到_ORDER

  //从文件中读出root节点的地址，并定位到该地址，然后读出node节点并放到db->path[0]中，同时更新db->node_addrs[0]为root节点的地址
  fseeko(db->fp, 0, SEEK_SET); //定位到文件开头的位置，文件大小用off_t来表示，应该是64位
  fread(&r_addr, 1, SIZEOF_LONG, db->fp);
  fseeko(db->fp, r_addr, SEEK_SET);
  fread(db->path[index], 1, _WIDTH, db->fp);
  db->node_addrs[index] = r_addr;
//从root节点开始，往下遍历B+树查找key所在的位置，并将查找路径上的节点放到db->path[]中，并将找到这些节点的路径放到db->node_addrs[]中
search:
  isleaf = db->path[index][0]; //node中的第一个字节为0或者1表示是否为叶子节点
  //在当前node中遍历。因为key定长，指针也是定长，这俩组合可以视作一项，i这个游标每次移动一项
  for (; i < _WIDTH; i += (_HASH + SIZEOF_LONG))
  {
    //如果当前node中的当前key1和待查找的key相同，则说明找到了
    if (!strncmp(db->path[index] + i, key, _HASH))
    {
      //如果是叶子节点，那就是找到了（因为B+树的内部节点的key可能和叶子节点的key一样）
      if (isleaf)
      {
        *r_index = index;                                //更新查找的深度作为返回值
        i -= SIZEOF_LONG;                                //移动i到指向该key对应内容的地址
        uint64_t cindex = from_big(db->path[index] + i); //获取存放内容的地址content_index
        fseeko(db->fp, cindex, SEEK_SET);
        //下面这一部分应该是将content读出去，至于多长，需要从某个地方获取。
        unsigned char check;
        fread(&check, 1, 1, db->fp); //读取1个字节。这里只是一个原型，用于验证读到的内容是否正确？
        if (check == 0)
        {
          return 1;
        }
        return 0;
      }

      //下面这个应该不会存在，index如果超过了最大深度，应该直接退出，而不用比较。如果index已经超过了最大深度，怎么还存在key和待查找的key相等的情况呢？
      if (index >= _DEPTH)
      {
        *r_index = 0;
        return -1;
      }
      //不是叶子节点，而是内部节点，则需要找其右子树啦。找到右子树的根节点，并将该node放到db->path[]中，将其地址放到db->node_addrs[]中
      i += _HASH;
      uint64_t addr = from_big(db->path[index] + i);
      fseeko(db->fp, addr, SEEK_SET);
      ++index;
      fread(db->path[index], 1, _WIDTH, db->fp);
      db->node_addrs[index] = addr;
      i = SIZEOF_LONG + 1;
      goto search;
    }

    //如果遍历的key1比待查找的key大，或者key1为0（本node只有这么多项啦，这是最后一项），都说明key不在本层，
    //可以往下遍历左子树或者结束（如果到了叶子节点这一层）。这个if用于控制跨层
    if (strncmp(db->path[index] + i, key, _HASH) > 0 ||
        db->path[index][i] == 0)
    {
      //如果当前node是叶子节点，则说明key在当前的B+树中不存在，当前B+树的深度为index。如果后面要插入节点，则插入到本node中。
      //前面查找的过程以及记录了路径和路径上的节点，至于真的插入的时候是否会引起分裂，则插入之后再看。
      if (isleaf)
      {
        *r_index = index;
        return 1;
      }
      //如果超过最大深度了，则说明整棵B+树满了，或者出错了。正常情况下，遍历过程中是不会超出最大深度的。
      if (index >= _DEPTH)
      {
        *r_index = 0;
        return -1;
      }
      //当前node不是叶子节点，则需要往下找左子树，而i当前指向了key，所以需要往前移动8字节，才能达到记录左子树地址的位置。
      i -= SIZEOF_LONG;
      uint64_t addr = from_big(db->path[index] + i); //地址为啥用大端表示？查找过程应该不会涉及地址的比较
      //读取左子树的根节点，并将它放到db->path[]中，并将左子树的根节点的地址放到db->node_addrs[]中。
      fseeko(db->fp, addr, SEEK_SET);
      ++index;
      fread(db->path[index], 1, _WIDTH, db->fp);
      db->node_addrs[index] = addr;

      i = SIZEOF_LONG + 1; //i定位到node中的第一个项的key的位置，开始在本node中查找
      goto search;
    }
  }
}

/**
 * 在B+树中插入一项key和value.并将相关信息保留在db中。
 * 我猜测文件的构成应该是开头8个字节表示root节点所在的位置，后面跟着一系列节点和记录（value）。
 * 每个记录包含1个字节的有效位，key_len和key构成的key字段，value_len和value构成的value字段.
 * B+树在使用的时候是从文件中读取出来（根据root节点及其左右子树地址），然后在内存中构建的，B+树本身也存到文件中了。
 */
void putkeyvalue(db *db, unsigned char *key, unsigned char *value)
{
  int index;
  uint64_t ret;
#ifdef LOCK
  if (db_lock(db) == -1)
  {
    perror("fcntl");
    return;
  }
  else
  {
#endif
    if ((ret = search(db, key, &index)) > 0) //key不存在，且找到了待插入的位置为合法位置
    {
      uint64_t k_len = strlen(key);
      uint64_t v_len = strlen(value);
      if (k_len + v_len > _MAX)
      {
        return;
      }
      uint64_t n_len = k_len + v_len + SIZEOF_LONG + SIZEOF_LONG + 1; //用于存储value的空间，格式为"0/1 key_len key value_len value".

      unsigned char *nnode = malloc(n_len + 1); //new node
      unsigned char *ptr = nnode;
      memset(nnode, 0, n_len + 1);
      nnode[0] = 1;                                                       //先填充有效位
      to_big(ptr + 1, k_len);                                             //先填充key_len(大端表示)
      strncpy(ptr + SIZEOF_LONG + 1, key, k_len);                         //再填充key
      to_big(ptr + SIZEOF_LONG + k_len + 1, v_len);                       //填充value_len
      strncpy(ptr + SIZEOF_LONG + k_len + SIZEOF_LONG + 1, value, v_len); //填充value
      fseeko(db->fp, 0, SEEK_END);                                        //定位到文件末尾处
      uint64_t addr = ftello(db->fp);                                     //获取当前位置
      fwrite(nnode, 1, n_len, db->fp);                                    //将前面构建的value结构写入到文件
      _insert(db, key, index, addr, 0, 1);                                //key和addr构成的<k,v>插入到B+树中
    }
#ifdef LOCK
    if (db_unlock(db) == -1)
    {
      perror("fcntl");
      return;
    }
  }
#endif
  fflush(db->fp);
}

char *
getvalue(db *db, unsigned char *key)
{
  int index;
  if (_HASH > strlen(key)) //先将key补齐到_HASH大小，后面填充'a'，在插入的时候估计也是把key先补齐到_HASH大小了
  {
    unsigned char *okey = key;
    key = malloc(_HASH + 1);
    memset(key, 0x61, _HASH + 1);
    strncpy(key, okey, strlen(okey));
  }

  if (!search(db, key, &index)) //如果找到了
  {
    int i = SIZEOF_LONG + 1;
    for (; i < _WIDTH; i += (SIZEOF_LONG + _HASH)) //在db->path[index]这个node内遍历查找key对应的项item
    {
      if (!strncmp(db->path[index] + i, key, _HASH)) //找到了
      {
        i -= SIZEOF_LONG;
        uint64_t addr = from_big(db->path[index] + i); //找到key对应的addr，该addr记录了对应的内容在文件中的位置。
        fseeko(db->fp, addr, SEEK_SET);
        unsigned char exists;
        unsigned char k_len[SIZEOF_LONG];
        unsigned char v_len[SIZEOF_LONG];
        fread(&exists, 1, 1, db->fp);
        if (!exists) //该记录无效
        {
          return NULL;
        }
        //该记录有效，读取key和value
        fread(k_len, 1, SIZEOF_LONG, db->fp);
        uint64_t k_lenb = from_big(k_len);
        unsigned char *k = malloc(k_lenb);
        memset(k, 0, k_lenb);
        fread(k, 1, k_lenb, db->fp);
        fread(v_len, 1, SIZEOF_LONG, db->fp);
        uint64_t v_lenb = from_big(v_len);
        unsigned char *v = malloc(v_lenb);
        memset(v, 0, v_lenb);
        fread(v, 1, v_lenb, db->fp);
        return v; //返回value
      }
    }
    return NULL;
  }
  return NULL;
}

/**
*delete并不真实删除数据，只是把记录的第一个字节从1变成了0，表示数据无效。
*/
void delete (db *db, unsigned char *key)
{
  int index;
  if (_HASH > strlen(key))
  {
    unsigned char *okey = key;
    key = malloc(_HASH + 1);
    memset(key, 0x61, _HASH + 1);
    strncpy(key, okey, strlen(okey));
  }
  if (!search(db, key, &index))
  {
    int i = SIZEOF_LONG + 1;
    for (; i < _WIDTH; i += (SIZEOF_LONG + _HASH))
    {
      if (!strncmp(db->path[index] + i, key, _HASH))
      {
        i -= SIZEOF_LONG;
        unsigned char del = 0;
        uint64_t addr = from_big(db->path[index] + i);
        fseeko(db->fp, addr, SEEK_SET);
        fwrite(&del, 1, 1, db->fp);
      }
    }
  }
}

void db_init(db *db, const char *name)
{
  uint64_t addr;
  unsigned char *zero = malloc(_WIDTH);
  memset(zero, 0, _WIDTH);
  zero[0] = 1;
  db->fp = fopen(name, "rb+"); //尝试读写的方式打开一个确定存在的文件，如果文件不存在则打开失败，返回null。
  if (!db->fp)
  { //如果能打开，说明文件已经存在（但是无法判断是否是存储B+树的内容）
    db->fp = fopen(name, "wb+");
    addr = SIZEOF_LONG; //8
    fseeko(db->fp, 0, SEEK_SET);
    fwrite(&addr, SIZEOF_LONG, 1, db->fp); //写文件头的8个字节，内容是08 00 00 00 00 00 00 00 （小端）
    fwrite(zero, 1, _WIDTH, db->fp);       //写文件的第9个字节至8+_WIDTH字节，共计_WIDTH字节，写一个空的叶子node，（node中的第一个字节为1，表示叶子节点）
  }
}

void db_close(db *db)
{
  fclose(db->fp);
}

/*** function for testing，可以删除***/

unsigned char *random_str()
{
  int i;
 unsigned char *alphabet = "abcdefghijklmnopqrstuvwxyz";
 unsigned char *string = malloc(_HASH); //上面的_HASH为_HASH是测试用的，和这里一致
  for (i = 0; i < _HASH - 1; i++)
  {
    string[i] = alphabet[rand() % 26];
  }
  string[i] = 0;
  return string;
}

/***************************/
