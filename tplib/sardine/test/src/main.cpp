#include <gtest/gtest.h>

#include <sardine/config.hpp>
#include <config.hpp>

#include <boost/interprocess/shared_memory_object.hpp>
#include <sardine/sardine.hpp>

struct Environment : ::testing::Environment
{
    // Override this to define how to set up the environment.
    void SetUp() override
    {
        using namespace sardine;

        cache::clear();

        boost::interprocess::shared_memory_object::remove(host_filename);
        boost::interprocess::shared_memory_object::remove(host_filename_2);
        boost::interprocess::shared_memory_object::remove(shm_filename);
        boost::interprocess::shared_memory_object::remove(managed_filename);
    }

    // Override this to define how to tear down the environment.
    void TearDown() override
    {
        using namespace sardine;

        cache::clear();

        boost::interprocess::shared_memory_object::remove(host_filename);
        boost::interprocess::shared_memory_object::remove(host_filename_2);
        boost::interprocess::shared_memory_object::remove(shm_filename);
        boost::interprocess::shared_memory_object::remove(managed_filename);
    }
};


int main(int argc, char* argv[]) {
    ::testing::InitGoogleTest(&argc, argv);
    // gtest takes ownership of the TestEnvironment ptr - we don't delete it.
    ::testing::AddGlobalTestEnvironment(new ::Environment);
    return RUN_ALL_TESTS();
}
