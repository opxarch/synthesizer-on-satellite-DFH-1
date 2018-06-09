
#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <stdbool.h>
#include <assert.h>

#include <math.h>

#include "fft.h"

typedef struct
{
   char gate;
   float freq; /* Hz */
   float pals; /* omega = 2pi * f */
   float time, tenv;
   float phase;
   float attack;
   float decay;
   float decay_k;
   float sustain;
   float tremolo;
   float tremolo_k;
   float delta_s;
} sample_gen_t;

typedef struct /* Chebyshev filter coefficients */
{
   double *a;
   double *b;
} flt_coeff_t;

static sample_gen_t  units[9];
static float         max_level         = 11024.0f;
static float         delta_t           = .0f;
static float         delta_s           = .0f;
static float         delta_a           = .0f;
static int           sample_rate       = 44100; /* samples/s (sps) */
static float         attack_time       = 0.06f; /* s */
static float         decay_threshold   = 0.74f;
static float         sustain_time      = 1.5f;
static float         sustain_accel     = 0.0000082f; /* sustain curve acceleration */
static float         tempi_time        = 0.15f;
static float         tremolo_threshold = 0.2f;
static float         tremolo_min       = 0.005f;
static float         tremolo_freq      = 5.0f; /* Hz */
static float         freq_error        = 0.0005f; /* analog-like frequency error factor */
static bool          stop_note_gate    = false;
static float         pitch             = .30f;
static float         cutoff            = 3000.0f; /* Cutoff frequency */
static double        filter_pr         = 0.6f; /* percent ripple */
static int           filter_np         = 6; /* the number of poles */
static flt_coeff_t   filter_coeffs     = {0};
static bool          filter_gate       = true;

/* struct to hold return values from chebyshev_poles function */
typedef struct {
    double a0;
    double a1;
    double a2;
    double b1;
    double b2;
} POLES;

static float freq_table[] = { /* based on twelve-tone temperament. */
   /* f2 (1) */ 698.46,
   /* g2 (2) */ 783.99,
   /* a2 (3) */ 880.0,
   /* c3 (5) */ 1046.50,
   /* d3 (6) */ 1174.66,
   /* f3 (#1)*/ 1396.91,
   /* e2 (b7)*/ 659.26,
   /* d2 (b6)*/ 587.33,
   /* c2 (b5)*/ 523.25,
};

#ifndef M_PI
#define M_PI (3.1415926)
#endif

typedef struct
{
   int genid; /* Hz */
   int duration;
} note_t;

static note_t note_sequence[] = {
   { 3, 8 },   /* 5 */
   { 3, 4 },   /* _5_ */
   { 4, 4 },   /* _6_ */
   { 1, 16 },  /* 2 - */
   { 0, 0 },
   { 0, 8 },   /* 1 */
   { 0, 4 },   /* _1_ */
   { 7, 4 },   /* _6_ */
   { 1, 16},   /* 2 - */
   { 0, 0 },
   { 3, 8 },   /* 5 */
   { 3, 8 },   /* 5 */
   { 4, 4 },   /* _6_ */
   { 5, 4 },   /* _#1_ */
   { 4, 4 },   /* _6_ */
   { 3, 4 },   /* _5_ */
   { 0, 8 },   /* 1 */
   { 0, 4 },   /* _1_ */
   { 7, 4 },   /* _b6_ */
   { 1, 16},   /* 2 - */
   { 0, 0 },
   { 0, -1},   /* goto 0 */
   { 3, 8 },   /* 5 */
   { 1, 8 },   /* 2 */
   { 0, 8 },   /* 1 */
   { 6, 4 },   /* _b7_ */
   { 7, 4 },   /* _b6_ */
   { 8, 8 },   /* b5 */
   { 3, 8 },   /* 5 */
   { 1, 8 },   /* 2 */
   { 2, 4 },   /* _3_ */
   { 1, 4 },   /* _2_ */
   { 0, 8 },   /* 1 */
   { 0, 4 },   /* _1_ */
   { 7, 4 },   /* _b6_ */
   { 1, 4 },   /* _2_ */
   { 2, 4 },   /* _3_ */
   { 1, 4 },   /* _2_ */
   { 0, 4 },   /* _1_ */
   { 1, 4 },   /* _2_ */
   { 0, 4 },   /* _1_ */
   { 6, 4 },   /* _b7_ */
   { 7, 4 },   /* _b6_ */
   { 8, 20}    /* b5 - b5 */
};

/**
 * Calculates the poles struct for chebyshev function
 * @param fc   filter cutoff (0 to 0.5 percent of sampling frequency)
 * @param lp     false for low pass, true for high pass
 * @param pr   Percent Ripple (0 to 29)
 * @param np      Number of poles (2 to 20, must be even)
 * @param p       Current pole being calculated
 */
POLES
chebyshev_poles( double fc, bool lp, double pr, int np, int p )
{
   /* calculate the location of the pole on the unit circle */
   double angle = M_PI/(np*2) + (p-1)*M_PI/np;
   double rp = - cos(angle);
   double ip =   sin(angle);

   double es = 0;
   double vx = 0;
   double kx = 0;
   
   if (pr != 0.0) { /* warp from circle to eclipse */
     double temp = 100/(100 - pr);
     es = sqrt(temp*temp - 1);
     vx = log((1/es) + sqrt(1/(es*es) + 1))/np;

     kx = log((1/es) + sqrt(1/(es*es) - 1))/np;
     kx = (exp(kx) + exp(-kx)) / 2;
     
     rp *= (exp(vx) - exp(-vx))/(2*kx);
     ip *= (exp(vx) + exp(-vx))/(2*kx);
   } 

   /* s domain to z domain transform */
   double t = 2 * tan(1.0/2.0);
   double w = 2 * M_PI * fc;
   double m = rp*rp + ip*ip;
   double d = 4 - 4*rp*t + m*t*t;

   double x0 = t*t/d;
   double x1 = 2*x0;
   double x2 = x0;
   double y1 = (8 - 2*m*t*t)/d;
   double y2 = (-4 - 4*rp*t - m*t*t)/d;

   /* LP to LP or LP to HP transform */
   double k;
   if (lp) {
     k = - cos(w/2 + 1.0/2.0) / cos(w/2 - 1.0/2.0);
   } else {
     k = sin(1.0/2.0 - w/2) / sin(1.0/2.0 + w/2);
   }

   d = 1 + y1*k - y2*k*k;

   POLES ret;
   ret.a0 = (x0 - x1*k + x2*k*k) / d;
   ret.a1 = (-2*x0*k + x1 + x1*k*k - 2*x2*k) / d;
   ret.a2 = (x0*k*k - x1*k + x2) / d;
   ret.b1 = (2*k + y1 + y1*k*k - 2*y2*k) / d;
   ret.b2 = (-k*k - y1*k + y2) / d;

   if (lp) {
     ret.a1 = -ret.a1;
     ret.b1 = -ret.b1;
   }

   return ret;
}

/**
  * Chebyshev filter coefficients
  * @param fc   filter cutoff (0 to 0.5 percent of sampling frequency)
  * @param lp     false for low pass, true for high pass
  * @param pr   Percent Ripple (0 to 29)
  * @param np      Number of poles (2 to 20, must be even)
  * @param *a   output buffer to hold a coefficients
  * @param *b   output buffer to hold b coefficients
  */
flt_coeff_t
chebyshev( float fc, bool lp, double pr, int np )
{
   int n = 23; /* size of all arrays */
   flt_coeff_t coeff;
   
   double ta[n];
   double tb[n];
   
   coeff.a = (double *)malloc( sizeof(double) * n );
   coeff.b = (double *)malloc( sizeof(double) * n );
   
   for (int i = 0; i < n; ++i) {
     coeff.a[i] = 0;
     coeff.b[i] = 0;
   }
   coeff.a[2] = 1;
   coeff.b[2] = 1;

   for (int p = 1; p < np/2+1; ++p) { /* each pole pair */
     POLES ret = chebyshev_poles(fc, lp, pr, np, p);

     /* add coefficients to the cascade */
     for (int i = 0; i < n; ++i) {
         ta[i] = coeff.a[i];
         tb[i] = coeff.b[i];
     }

     for (int i = 2; i < n; ++i) {
         coeff.a[i] = ret.a0*ta[i] + ret.a1*ta[i-1] + ret.a2*ta[i-2];
         coeff.b[i] =        tb[i] - ret.b1*tb[i-1] - ret.b2*tb[i-2];
     }
   }

   /*
    finish combining coefficients
    */
   coeff.b[2] = 0;
   for (int i = 0; i < n-2; ++i) {
     coeff.a[i] =  coeff.a[i+2];
     coeff.b[i] = -coeff.b[i+2];
   }

   /*
    normalize the gain
    */
   double sa = 0;
   double sb = 0;
   for (int i = 0; i < n; ++i) {
     if (!lp) {
         sa += coeff.a[i];
         sb += coeff.b[i];
     }
     else {
         sa += coeff.a[i]*pow(-1,i); 
         sb += coeff.b[i]*pow(-1,i);
     }
   }
   double gain = sa/(1-sb);
   for (int i = 0; i < n; ++i)
     coeff.a[i] /= gain;

   return coeff;
}

/**
 * Applies filter to input buffer and stores in output buffer
 * Calculates the convolution of in by a, along with rescursive feedback from b
 * @param *in               input buffer
 * @param in_n                 input buffer length
 * @param *a                filter kernel
 * @param *b                filter kernel (recursive part)
 * @param flt_n                filter kernel length (assumes equal length for a and b)
 * @param *out              output buffer
 * @param out_n                output buffer length
 */
void
apply_filter( double *in, int in_n, double *a, double *b, int flt_n, double *out, int out_n )
{
   for (int i = 0; i < out_n; ++i) {
     out[i] = 0;
     for (int j = 0; j < flt_n; ++j) {
         double atemp = 0;
         double btemp = 0;
         if (i - j >= 0) {
             btemp = b[j]*out[i-j];
             if (i - j < in_n) {
                 atemp = a[j]*in[i-j];
             }
         }
         out[i] += atemp + btemp;
     }
   }
}

static int16_t
getsample( int genid )
{
   double sample;
   sample_gen_t *gen = &units[genid];
   
   switch ( gen->gate )
   {   
      case 3:
         /* update the sustain envelop */
         if ( gen->sustain >= .0f ) {
            gen->sustain -= (gen->delta_s);
            gen->delta_s -= gen->delta_s * sustain_accel;
            
         } else {
            gen->sustain = .0f;
            gen->decay = 1.0f;
            gen->attack = .0f;
            gen->gate = 0;
         }
         
         gen->tremolo = .0f;//tremolo_min;
         
      case 2:
         /* update the decay envelop */
         if ( gen->attack > 1.0f )
            gen->attack -= delta_a * 2.0f;
         else {
            gen->attack = 1.0f;
            if ( gen->decay >= decay_threshold ) {
               gen->decay -= gen->decay_k;
            }
         }
         
      case 1: {
         
         /*
          * simulating the analog-like frequency error
          */
         float freq_error_factor = (1.0f - (1.0f + sin( M_PI * 2.0f * 2.0f * gen->tenv + gen->phase )) * freq_error);
         float phase = gen->pals * ( ( gen->gate == 1 ) ? freq_error_factor : 1.0f ) * pitch * gen->time + gen->phase;

         /*
          * sawtooth function
          */
         if ( phase <= M_PI )
            sample = (2.0f / M_PI * phase - 1.0f) * max_level;
         else
            sample = (- (2.0f / M_PI) * phase + 3.0f) * max_level;
         
         /* extra low-frequency harmonic */
         sample *= 1.0 - (1.0 + sin(gen->pals * ( ( gen->gate == 1 ) ? freq_error_factor : 1.0f ) * gen->tenv + gen->phase)) * 0.5;
         
#if 0 /* test harmonic */
         sample += sin(phase) * max_level * 0.1;
         sample *= 1.0 - (1.0 + sin(phase)) * 0.5;
         sample *= sin( gen->pals * gen->time + gen->phase ) ;
         sample *= cos( gen->pals /2 * gen->time + gen->phase + M_PI / 3 );
#endif

         /*
          * amplitude envelope generator
          */
         sample *= 1.0f - (1.0f + sin( M_PI * 2.0f * tremolo_freq * gen->tenv + gen->phase )) * gen->tremolo;
         sample *= gen->decay;
         sample *= gen->attack;
         sample *= gen->sustain;

         if ( gen->tremolo <= tremolo_threshold && gen->gate != 3  )
            gen->tremolo += gen->tremolo_k;
         
         if ( phase <= (2.0f * M_PI - delta_t) )
            gen->time += delta_t;
         else
            gen->time = .0f;
         gen->tenv += delta_t;
         
         /* update the attack envelop */
         if ( gen->attack < 1.2f ) {
            gen->attack += delta_a;
         } else {
            gen->gate = 2;
         }
         
         /* pitch shifter ( process from circuit powering on to achieving stability ) */
         if ( pitch < 1.0f )
            pitch += 0.0006f;
         else
            pitch = 1.0f;
         
         return (int16_t) sample;
      }
      
      default:
         return 0; /* gate off */
   }
   return 0;
}

static void
init_genunit( int genid, float freq, float phase )
{
   sample_gen_t *gen = &units[genid];
   
   gen->gate = 0;
   gen->freq = freq;
   gen->pals = M_PI * 2.0f * freq;
   gen->time = .0f;
   gen->phase = phase;
   gen->decay = 1.0f;
   gen->attack = .0f;
   gen->sustain = .0f;
   gen->decay_k = .0f;
   gen->tenv = .0f;
   gen->tremolo = tremolo_min;
   gen->tremolo_k = .0f;
   gen->delta_s = .0f;
}

static void
gate_on( int genid, float duration )
{
   units[genid].gate = 1;
   units[genid].decay = 1.0f;
   units[genid].attack = .2f;
   units[genid].sustain = 1.0f;
   units[genid].tenv = .0f;
   units[genid].tremolo = tremolo_min;
   units[genid].delta_s = delta_s;
   
   /* calc the decay and tremolo parameter that is based on the duration */
   units[genid].decay_k = (1.0f - decay_threshold) / duration * delta_t;
   units[genid].tremolo_k = (tremolo_threshold - tremolo_min) / (duration + 0.1f) * delta_t;
}

static void
gate_sustain( int genid ) {
   units[genid].gate = 3;
}

static void
stop_tone( void )
{
   for ( int i = 0; i < sizeof(units) / sizeof(*units); i++ ) {
      units[i].gate = 0;
      units[i].decay = .0;
   }
}

/* The volume ranges from 0 - 128 */
#define ADJUST_LEVEL(s, v)	(s = (s*v)/MIXER_MAX_LEVEL)
#define ADJUST_LEVEL_U8(s, v)	(s = (((s-128)*v)/MIXER_MAX_LEVEL)+128)
#define MIXER_MAX_LEVEL (128)

/**
 * Mix the two audio stream.
 * @param dst       Pointer to the target buffer.
 * @param src       Pointer to the source buffer.
 * @param nsamples  The number of samples.
 * @param volume    The volume the mixer limited.
 * @return status code.
 */
int
mix_samples (int16_t *dst, const int16_t *src, uint32_t nsamples, int volume)
{
  if ( volume == 0 )
    {
      return -1;
    }

  int64_t src1, src2;
  int64_t dst_sample;
  const int64_t max_audioval = ((1<<(sizeof(int16_t)*8-1))-1);
  const int64_t min_audioval = -(1<<(sizeof(int16_t)*8-1));

  while ( nsamples-- )
    {
      /*
       * synth
       */
      src1 = *src;
      ADJUST_LEVEL(src1, volume);
      src2 = *dst;
      src++;
      dst_sample = src1 + src2;

      /*
       * clip ?
       */
      if ( dst_sample > max_audioval )
        {
          dst_sample = max_audioval;
        }
      else
      if ( dst_sample < min_audioval )
        {
          dst_sample = min_audioval;
        }

      *dst = dst_sample;
      dst++;
    }

  return 0;
}

#define MAX_BUF_SIZE 8192
#define DUMP_FFT 1

static void
fft_buff( double *buff_in, int nsmpls, float *proc_mag, float *proc_phase )
{
   int pos = 0;
   float work_rex[MAX_BUF_SIZE];
   float work_imx[MAX_BUF_SIZE];
   
   double rex, imx, mag, phase, window;
   int N = nsmpls, N2, i = 0;

   assert( N <= MAX_BUF_SIZE );
   
   N2 = N / 2;
   window = 1.0f;

   for (i = 0; i < N; i++) {      
      window = -0.5 * cos(2.0 * M_PI * i / N) + 0.5;
      work_rex[i] = window * buff_in[pos++];
      work_imx[i] = 0;
   }

   fft( work_rex, work_imx, N );

   for (i = 0; i < N2; i++) {

      rex = work_rex[i];
      imx = work_imx[i];

      if ( proc_mag ) {
         mag = 2.0 * sqrt(rex * rex + imx * imx);
         proc_mag[i] = mag;
      }
      
      if ( proc_phase ) {
         phase = atan2(imx, rex);
         proc_phase[i] = (float) phase;
      }
   }
}

int
main( int argc, char *argv[] )
{
   FILE *fp;
   int num_unit = sizeof(units) / sizeof(*units);
   
   (void)argc;
   (void)argv;
   
   /*
    * calc the parameters
    */
   delta_t = 1.0 / sample_rate;
   delta_s = 1.0 / sustain_time * delta_t;
   delta_a = 1.0 / attack_time * delta_t;
   
   for ( int i = 0; i < num_unit; i++ ) {
      init_genunit( i, freq_table[i], .0);
   }
   
   filter_coeffs = chebyshev( cutoff / sample_rate, false, filter_pr, filter_np );
   
   printf("the number of units = %d\n", num_unit);
   printf("the number of note_sequence = %d\n", sizeof(note_sequence) / sizeof(*note_sequence));
   printf("sample_rate = %d\n", sample_rate);
   printf("sustain_time = %f\n", sustain_time);
   printf("delta_t = %f\n", delta_t);
   printf("delta_s = %f\n", delta_s);
   printf("delta_a = %f\n", delta_a);

   /*
    * create/open the target file to write
    */
   if ( !( fp = fopen( "output.pcm", "wb+" ) ) ) {
      fprintf( stderr, "unable to open target file.\n" );
      return 1;
   }
   
   int16_t sample = 0, buff = 0;
   float duration = .0f;
   int note_id = 0, last_note = -1;
   
   for ( ;; ) {
      /*
       * process the duration of each note
       */
      if ( duration <= 0 ) {
         if ( last_note > -1 ) gate_sustain( last_note );
         if ( note_id >= sizeof(note_sequence) / sizeof(*note_sequence) ) {
            for ( int i = 0; i < num_unit; i++ ) {
               if ( units[i].sustain > .0 )
                  goto sust; /* wait for the sustain */
            }
            printf("completed\n");
            break;
         }
         if ( note_sequence[note_id].duration > 0 ) {
            int genid = note_sequence[note_id].genid;
            duration = tempi_time * note_sequence[note_id].duration;
            
            gate_on( genid, duration ); /* trigger the note */
            
         } else if ( note_sequence[note_id].duration < 0 ) {
            int goto_id = -note_sequence[note_id].duration - 2;
            if ( goto_id >= 0 || goto_id < num_unit ) {
               note_sequence[note_id].duration = 0;
               note_id = goto_id; /* goto the specified note */
            }
         } else if( stop_note_gate )
            stop_tone();
         
         /* we're only processing single note */
         last_note = note_sequence[note_id].genid;
         note_id++;
      }
      
sust:
      buff = .0;
      
      /*
       * Get samples from all the oscillators and then mix the audio
       */
      for ( int j = 0; j < num_unit; j++ ) {
         sample = getsample( j );
         mix_samples( &buff, &sample, 1, MIXER_MAX_LEVEL );
      }
      
#if 0
      printf("sample = %d\n", buff);
#endif

      if ( !fwrite( &buff, sizeof(sample), 1, fp ) || ferror( fp ) ) {
         fprintf( stderr, "unable to write file.\n" );
         return 1;
      }
      
      duration -= delta_t;
   }
   
   size_t len = ftell( fp );
   
   /*
    * filter the original samples
    */
   if ( filter_gate ) {
      int pos = 0;
      int sample_num = len / sizeof(int16_t);
      double *flt_in = (double *)malloc( sample_num * sizeof(*flt_in) );
      double *flt_out = (double *)malloc( sample_num * sizeof(*flt_out) );
      float fft_in_mag[MAX_BUF_SIZE / 2];
      float fft_out_mag[MAX_BUF_SIZE / 2];
      int fft_nsmpl = 1024, fft_nsmpl2;

      if ( !flt_in || !flt_out ) {
         fprintf( stderr, "no memory!\n" );
         return 1;
      }
      printf("processing...\n");
      printf( "sample_num = %d\n", sample_num );
      
      fflush( fp );
      
      /* this mess is to read from the original wave, apply chebyshev filter and then write back */
      fseek( fp, 0, SEEK_SET );
      len = sample_num;
      while ( len-- ) {
         fread( &sample, sizeof(int16_t), 1, fp );
         flt_in[pos++] = (double) sample;
      }
      
      fft_buff( flt_in, fft_nsmpl, fft_in_mag, NULL );
      
      apply_filter( flt_in, sample_num, filter_coeffs.a, filter_coeffs.b, filter_np+1, flt_out, sample_num );
      
      fft_buff( flt_out, fft_nsmpl, fft_out_mag, NULL );
      
#if DUMP_FFT
   {
      int i;
      FILE *fplot = fopen( "fft_output.txt", "w" );
      
      fft_nsmpl2 = fft_nsmpl / 2;
      
      for ( i = 0; i < fft_nsmpl2; i++ ) fprintf( fplot, "%f ", fft_in_mag[i] );
      fprintf( fplot, "\n" );
      for ( i = 0; i < fft_nsmpl2; i++ ) fprintf( fplot, "%f ", fft_out_mag[i] );
      
      fclose( fplot );
   }
#else
      (void) fft_in_mag;
      (void) fft_out_mag;
      (void) fft_buff;
#endif
      
      fseek( fp, 0, SEEK_SET );
      len = sample_num;
      pos = 0;
      while ( len-- ) {
         sample = (int16_t) ( flt_out[pos++]);
         fwrite( &sample, sizeof(int16_t), 1, fp );
      }
   }
   
   fclose( fp );
   return 0;
}
