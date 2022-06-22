#pragma once
#include <vector>
#include <fstream>

#include <iostream>
using namespace std;

struct TreeNode
{
    uint32_t val;
    TreeNode *left;
    TreeNode *right;
    TreeNode() : val(0), left(nullptr), right(nullptr) {}
    TreeNode(uint32_t x) : val(x), left(nullptr), right(nullptr) {}
    TreeNode(uint32_t x, TreeNode *left, TreeNode *right) : val(x), left(left), right(right) {}
};
class CBTInserter
{
private:
    TreeNode *valsptr;
    TreeNode *temp;
   TreeNode *prev;//定位到待插入位置的前一个结点
public:
    TreeNode *root;
    vector<uint32_t> vals;
    uint32_t cap;
    CBTInserter(int cap) : cap(cap)
    {
        root = new TreeNode(0x7fffff);
        valsptr = (TreeNode *)calloc(cap, sizeof(TreeNode));
        vals.reserve(cap);
    }
    void reset()
    {
        memset(valsptr, 0, sizeof(TreeNode) * cap);
        vals.resize(0);
        root->left = nullptr;
        root->right = nullptr;
    }
    inline void insert(const uint32_t &key)
    {
        //定义一个临时指针 用于移动
       temp = root;    //方便移动 以及 跳出循环
        while (temp != nullptr)
        {
            prev = temp;
            if (key < temp->val)
            {
                temp = temp->left;
            }
            else if (key > temp->val)
            {
                temp = temp->right;
            }
            else
            {
                return;
            }
        }
        if (key < prev->val)
        {
            prev->left = valsptr + vals.size();
            prev->left->val = key;
            vals.push_back(key);
        }
        else
        {
            prev->right = valsptr + vals.size();
            prev->right->val = key;
            vals.push_back(key);
        }
    }
    void insert(uint32_t *p32, uint32_t key)
    {
        //定义一个临时指针 用于移动
        TreeNode *temp;
        TreeNode *prev; //定位到待插入位置的前一个结点

        for (register uint32_t i = 0; i < key; i++)
        {
        one:
            temp = root; //方便移动 以及 跳出循环
            while (temp != nullptr)
            {
                prev = temp;
                if (key < temp->val)
                    temp = temp->left;
                else if (key > temp->val)
                    temp = temp->right;
                else
                {
                    i++;
                    if (i >= key)
                        return;
                    goto one;
                }
            }
            if (key < prev->val)
            {
                prev->left = valsptr + vals.size();
                prev->left->val = key;
                vals.push_back(key);
            }
            else
            {
                prev->right = valsptr + vals.size();
                prev->right->val = key;
                vals.push_back(key);
            }
        }
    }
    TreeNode *get_root()
    {
        return root;
    }
};

// {
//     Timer t("sett");
//     std::set<uint32_t> set32(_32buf, _32buf + 640 * 480);
// }