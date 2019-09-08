/*
 *  dht.c:
 *	read temperature and humidity from DHT11 or DHT22 sensor
 */

#include <wiringPi.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <sys/time.h>

#define MAX_TIMINGS	85
#define DHT_PIN		7	/* GPIO-32 */
/* #define DEBUG */

int data[5] = { 0, 0, 0, 0, 0 };

void read_dht_data(unsigned int cntlevel)
{
	uint8_t laststate	= HIGH;
	uint8_t counter		= 0;
	uint8_t j			= 0, i;
    uint8_t cnth        = 0;
    uint8_t cntl        = 0;
    uint8_t ch          = 0;
    uint8_t cl          = 0;
    struct timeval tNow, tPrev;
#ifdef DEBUG
    float cnthigh, cntlow;
#endif

	data[0] = data[1] = data[2] = data[3] = data[4] = 0;

	/* pull pin down for 18 milliseconds */
	pinMode( DHT_PIN, OUTPUT );
	digitalWrite( DHT_PIN, LOW );
	delay( 10 );

	/* prepare to read the pin */
	pinMode( DHT_PIN, INPUT );

	/* detect change and read data */
	for ( i = 0; i < MAX_TIMINGS; i++ )
	{
		counter = 0;
        gettimeofday (&tPrev, NULL);
        
		while ( digitalRead( DHT_PIN ) == laststate )
		{
			counter++;
			delayMicroseconds( 1 );
			if ( counter == 255 )
			{
				break;
			}
		}
		laststate = digitalRead( DHT_PIN );
        gettimeofday (&tNow, NULL);

		if ( counter == 255 )
			break;

		/* ignore first 3 transitions */
		if ( (i >= 4) && (i % 2 == 0) )
		{
			/* shove each bit into the storage bytes */
			data[j / 8] <<= 1;
            
//			if ( counter > cntlevel )
            if ( (tNow.tv_sec-tPrev.tv_sec)*1000000+tNow.tv_usec-tPrev.tv_usec > 50.0 )
            {
				data[j / 8] |= 1;
                cnth=cnth+counter;
                ch++;
            }
            else
            {
                cntl=cntl+counter;
                cl++;
            }
			j++;
		}
	}
#ifdef DEBUG
    /*
     * get average high level and low level counter values
     */
    cnthigh=(float)cnth/(float)ch;
    cntlow=(float)cntl/(float)cl;
#endif
	/*
	 * check we read 40 bits (8bit x 5 ) + verify checksum in the last byte
	 * print it out if data is good
	 */
	if ( (j >= 40) &&
	     (data[4] == ( (data[0] + data[1] + data[2] + data[3]) & 0xFF) ) )
	{
		float h = (float)((data[0] << 8) + data[1]) / 10;
		if ( h > 100 )
		{
			h = data[0];	// for DHT11
		}
		float c = (float)(((data[2] & 0x7F) << 8) + data[3]) / 10;
		if ( c > 125 )
		{
			c = data[2];	// for DHT11
		}
		if ( data[2] & 0x80 )
		{
			c = -c;
		}
		float f = c * 1.8f + 32;
		printf( "Humidity = %.1f %% Temperature = %.1f *C (%.1f *F)\n", h, c, f );
        exit(0);
#ifdef DEBUG
        printf( "Low/High Level Average: %4.1f, %4.1f\n\n", cntlow, cnthigh);
#endif
	}
#ifdef DEBUG
    else  {
		printf( "Data not good (j=%d should be >=40), skip\n", j );
        if (j >= 40) {
            printf( "Data is %d, %d, %d, %d, %d\n", data[0], data[1], data[2], data[3], data[4]);
            printf( "Low/High Level Average: %4.1f, %4.1f\n\n", cntlow, cnthigh);
        }
	}
#endif
}

int main( void )
{
    
#ifdef DEBUG    
	printf( "NanoPi M4 DHT22 temperature/humidity test\n" );
#endif

	if ( wiringPiSetup() == -1 )
		exit( 1 );

	while ( 1 )
	{
		read_dht_data(25);
		delay( 2000 ); /* wait 2 seconds before next read */
	}

	return(0);
}