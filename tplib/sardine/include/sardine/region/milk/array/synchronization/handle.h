#ifndef CACAO_SYNC_HANDLE_H
#define CACAO_SYNC_HANDLE_H

#include <cacao/sync/guard.h>

#if EMU_CUDA
#include <emu/cuda.h>
#endif

namespace cacao
{

namespace sync
{

    struct handle_item
    {
        Array* array;
        Mode mode;
        SyncData s_data;
    };

    struct handle_t
    {
        std::vector<handle_item> items;

        void wait() {
            for (auto& item : items) {
                switch (item.mode)
                {
                    case Mode::internal:
                        item.array->wait();
                        break;
                    case Mode::guard:
                        // guard::wait_and_reset(*item.array, item.s_data);
                        throw std::runtime_error("Not implemented yet.");
                        break;
                    case Mode::stream:
                        throw std::runtime_error("Not supported: stream wait with CPU handle.");
                }
            }
        }

        void notify() {
            for (auto& item : items) {
                switch (item.mode)
                {
                    case Mode::internal:
                        item.array->notify();
                        break;
                    case Mode::guard:
                        // guard::set(*item.array, item.s_data, GuardValue::unlock);
                        throw std::runtime_error("Not implemented yet.");
                        break;
                    case Mode::stream:
                        throw std::runtime_error("Not supported: stream notify with CPU handle.");
                }
            }
        }

        void reset() {
            for (auto& item : items) {
                switch (mode)
                {
                    case Mode::internal:
                        item.array->reset();
                        break;
                    case Mode::guard:
                        // guard::set(*item.array, item.s_data, GuardValue::lock);
                        throw std::runtime_error("Not implemented yet.");
                        break;
                    case Mode::stream:
                        throw std::runtime_error("Not supported: stream notify with CPU handle.");
                }
            }
        }

    };

} // namespace sync

} // namespace cacao

#endif //CACAO_SYNC_HANDLE_H























//     struct Handle
//     {
//         Array & array;

//         Mode mode;

//         SyncData s_data;

//         void wait() const {
//             switch (mode)
//             {
//                 case Mode::internal:
//                     array.wait();
//                     break;
//                 case Mode::guard:
//                     guard::wait_and_reset(array, s_data);
//                     break;
//                 case Mode::stream:
//                     throw std::runtime_error("Not supported: stream wait with CPU handle.");
//             }
//         }

//         void notify() {
//             switch (mode)
//             {
//                 case Mode::internal:
//                     array.notify();
//                     break;
//                 case Mode::guard:
//                     guard::set(array, s_data, GuardValue::unlock);
//                     break;
//                 case Mode::stream:
//                     throw std::runtime_error("Not supported: stream notify with CPU handle.");
//             }
//         }

//         void reset() {
//             switch (mode)
//             {
//                 case Mode::internal:
//                     array.reset();
//                     break;
//                 case Mode::guard:
//                     guard::set(array, s_data, GuardValue::lock);
//                     break;
//                 case Mode::stream:
//                     throw std::runtime_error("Not supported: stream notify with CPU handle.");
//             }
//         }
//     };

// #if EMU_CUDA

//     struct HandleDevice
//     {
//         Array & array;

//         void wait(stream_cref_t stream) {
//             switch (mode)
//             {
//                 case Mode::internal:
//                     array.wait();
//                     break;
//                 case Mode::guard:
//                     guard::wait_and_reset(array, s_data, stream);
//                     break;
//                 case Mode::stream:
//                     // stream::wait_and_reset(array, s_data, stream);
//                     throw std::runtime_error("Not implemented: stream wait yet.");
//             }
//         }

//         void notify(stream_cref_t stream) {
//             switch (mode)
//             {
//                 case Mode::internal:
//                     array.notify();
//                     break;
//                 case Mode::guard:
//                     guard::set(array, s_data, GuardValue::unlock, stream);
//                     break;
//                 case Mode::stream:
//                     // stream::set(array, s_data, GuardValue::unlock, stream);
//                     throw std::runtime_error("Not implemented: stream notify yet.");

//             }
//         }

//         void reset(stream_cref_t stream) {
//             switch (mode)
//             {
//                 case Mode::internal:
//                     array.reset();
//                     break;
//                 case Mode::guard:
//                     guard::set(array, s_data, GuardValue::lock, stream);
//                     break;
//                 case Mode::stream:
//                     // stream::set(array, s_data, GuardValue::lock, stream);
//                     throw std::runtime_error("Not implemented: stream notify yet.");
//             }
//         }
//     };

// #endif
