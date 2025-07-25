/*
 * Copyright 2022 Intel Corporation.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "topology.h"

#include <stdio.h>
#include <unistd.h>

LogicalProcessorSet ambient_logical_processor_set()
{
    static bool warning_printed = []() {
        fprintf(stderr, "# WARNING: this OS does not support thread affinity. Results may be unreliable.\n");
        return true;
    }();
    (void) warning_printed;

    // assume all CPUs
    LogicalProcessorSet result = {};
    long n = sysconf(_SC_NPROCESSORS_ONLN);
    for (long i = 0; i < n; ++i)
        result.set(LogicalProcessor(i));
    return result;
}
