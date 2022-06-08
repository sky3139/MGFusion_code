
#ifndef _CPPMMAPLIB_MMAPLIB_H_
#define _CPPMMAPLIB_MMAPLIB_H_

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <stdexcept>

namespace mmaplib
{
    class mmap
    {
    public:
        mmap(const char *path);
        ~mmap();

        bool is_open() const;
        size_t size() const;
        const char *data() const;

    private:
        void cleanup();
        int fd_;
        size_t size_;
        void *addr_;
    };

    inline mmap::mmap(const char *path) : fd_(-1),
                                          size_(0),
                                          addr_(MAP_FAILED)
    {

        fd_ = open(path, O_RDONLY);
        if (fd_ == -1)
        {
            std::runtime_error("");
        }
        struct stat sb;
        if (fstat(fd_, &sb) == -1)
        {
            cleanup();
            std::runtime_error("");
        }
        size_ = sb.st_size;
        addr_ = ::mmap(NULL, size_, PROT_READ, MAP_PRIVATE, fd_, 0);
        if (addr_ == MAP_FAILED)
        {
            cleanup();
            std::runtime_error("");
        }
    }

    inline mmap::~mmap() { cleanup(); }
    inline bool mmap::is_open() const { return addr_ != MAP_FAILED; }

    inline size_t mmap::size() const { return size_; }

    inline const char *mmap::data() const { return (const char *)addr_; }

    inline void mmap::cleanup()
    {
        if (addr_ != MAP_FAILED)
        {
            munmap(addr_, size_);
            addr_ = MAP_FAILED;
        }

        if (fd_ != -1)
        {
            close(fd_);
            fd_ = -1;
        }
        size_ = 0;
    }

} // namespace mmaplib
#endif