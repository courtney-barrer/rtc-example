#include <gtest/gtest.h>

#include <config.hpp>

#include <boost/interprocess/shared_memory_object.hpp>
#include <sardine/sardine.hpp>

struct Environment : ::testing::Environment
{
    // Override this to define how to set up the environment.
    void SetUp() override
    {
        sardine::cache::clear();

        myboost::interprocess::shared_memory_object::remove(host_filename);
        myboost::interprocess::shared_memory_object::remove(host_filename_2);
        myboost::interprocess::shared_memory_object::remove(shm_filename);
        myboost::interprocess::shared_memory_object::remove(managed_filename);
    }

    // Override this to define how to tear down the environment.
    void TearDown() override
    {
        myboost::interprocess::shared_memory_object::remove(host_filename);
        myboost::interprocess::shared_memory_object::remove(host_filename_2);
        myboost::interprocess::shared_memory_object::remove(shm_filename);
        myboost::interprocess::shared_memory_object::remove(managed_filename);
    }
};


int main(int argc, char* argv[]) {
    ::testing::InitGoogleTest(&argc, argv);
    // gtest takes ownership of the TestEnvironment ptr - we don't delete it.
    ::testing::AddGlobalTestEnvironment(new ::Environment);
    return RUN_ALL_TESTS();
}
