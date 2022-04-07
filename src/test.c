#include "server.h"
#include <stdio.h>

int main()
{
    rand_init();

    for (int i = 0; i < 100; i++) {
        printf("%u ", rand_get_u16());
    }
    printf("\n");

    for (int i = 0; i < 100; i++) {
        printf("%u ", rand_get_u32());
    }
    printf("\n");

    for (int i = 0; i < 100; i++) {
        printf("%ld ", rand_get_i64());
    }
    printf("\n");

    for (int i = 0; i < 100; i++) {
        printf("%f ", rand_get_float());
    }
    printf("\n");

    for (int i = 0; i < 100; i++) {
        printf("%lf ", rand_get_double());
    }
    printf("\n");

    rand_clean();
}
