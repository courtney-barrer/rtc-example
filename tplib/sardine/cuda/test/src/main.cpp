#include <gtest/gtest.h>

#include <sardine/sardine.hpp>

struct Environment : ::testing::Environment
{
    // Override this to define how to set up the environment.
    void SetUp() override {
        sardine::cache::clear();
    }

    // Override this to define how to tear down the environment.
    void TearDown() override {
        sardine::cache::clear();
    }
};


int main(int argc, char* argv[]) {
    ::testing::InitGoogleTest(&argc, argv);
    // gtest takes ownership of the TestEnvironment ptr - we don't delete it.
    ::testing::AddGlobalTestEnvironment(new ::Environment);
    return RUN_ALL_TESTS();
}
