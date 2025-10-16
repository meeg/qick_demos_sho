import sys
from qick.qick import *
import time
import matplotlib.pyplot as plt

class AxisSignalGeneratorAMOV4(SocIp):
    bindto = ['user.org:user:axis_signal_gen_amo_v4:1.0']
    REGISTERS = {'memw_start_reg'   : 0}

    # Min/max order for frequency modulation.
    FMOD_MIN_ORDER = 0
    FMOD_MAX_ORDER = 5

    # Min/max order for amplitude modulation.
    AMOD_MIN_ORDER = 0
    AMOD_MAX_ORDER = 3

    # Ports for configure_connections.
    STREAM_IN_DMA_PORT  = "s0_axis"
    STREAM_IN_CTRL_PORT = "s1_axis"
    STREAM_OUT_PORT     = "m0_axis"
    STREAM_OUT_AUX_PORT = "m1_axis"

    # Flags.
    HAS_DMA     = False
    HAS_TPROC   = False
    HAS_DAC     = False
    HAS_AUX     = False
    
    def __init__(self, description):
        # Initialize ip
        super().__init__(description)
        
        # Default registers.
        self.memw_start_reg     = 0 # Don't write.

        # Generics.
        self.BT     = int(description['parameters']['BT'])
        self.NMEM   = int(description['parameters']['NMEM'])
        self.NDDS   = int(description['parameters']['NDDS'])
        self.NREG   = int(description['parameters']['NREG'])
        self.BFREQ  = int(description['parameters']['BFREQ'])
        self.BAMP   = int(description['parameters']['BAMP'])

        # Number of memory locations per DDS.
        self.MEM_LENGTH = 2**(self.NMEM)

    def configure(self):
        # Sampling period us.
        self.cfg['ts'] = 1/self.cfg['fs']
        
        # Frequency resolution (Q1.x).
        self.cfg['df']  = self.cfg['fs']/2**self.BFREQ

        # Sweep time.
        self.cfg['SWEEP_TIME_US'] = self.cfg['ts']*(2**(self.BT-1))

        # Phase resolution.
        self.dphi = 360/(2**self.BFREQ)

    def configure_connections(self, soc):
        self.soc = soc

        # Block type.
        self.cfg['type'] = soc.metadata.mod2type(self.fullpath)

        ##################################################
        ### Backward tracing: should finish at the DMA ###
        ##################################################
        ((block,port),) = soc.metadata.trace_bus(self.fullpath, self.STREAM_IN_DMA_PORT)

        while True:
            blocktype = soc.metadata.mod2type(block)

            if blocktype == "axi_dma":
                self.HAS_DMA = True
                self.dma = getattr(soc, block)
                break
            elif blocktype == "axis_clock_converter":
                ((block, port),) = soc.metadata.trace_bus(block, 'S_AXIS')
            elif blocktype == "axis_switch":
                self.switch = getattr(soc, block)
                self.switch_ch = int(port[1:3])
                ((block, port),) = soc.metadata.trace_bus(block, 'S00_AXIS')
            else:
                raise RuntimeError("falied to trace port for %s - unrecognized IP block %s" % (self.fullpath, block))

        ####################################################
        ### Backward tracing: should finish at the tProc ###
        ####################################################
        ((block,port),) = soc.metadata.trace_bus(self.fullpath, self.STREAM_IN_CTRL_PORT)

        while True:
            blocktype = soc.metadata.mod2type(block)

            if blocktype == "axis_tproc64x32_x8":
                self.HAS_TPROC = True
                self.cfg['tproc_ch'] = int(port[1])-1
                break
            elif blocktype == "axis_cdcsync_v1":
                # Port name for back-tracing.
                pp = "s{}_axis".format(port[1])
                ((block, port),) = soc.metadata.trace_bus(block, pp)
            else:
                raise RuntimeError("falied to trace port for %s - unrecognized IP block %s" % (self.fullpath, block))

        #################################################
        ### Forward tracing: should finish on the DAC ###
        #################################################
        ((block,port),) = soc.metadata.trace_bus(self.fullpath, self.STREAM_OUT_PORT)

        while True:
            blocktype = soc.metadata.mod2type(block)

            if blocktype == "usp_rf_data_converter":
                self.HAS_DAC = True
                dac = port[1:3]
                self.cfg['dac'] = dac
                self.cfg['fs']  = soc.dacs[dac]['fs']/16
                break
            else:
                raise RuntimeError("falied to trace port for %s - unrecognized IP block %s" % (self.fullpath, block))

        #################################################
        ### Forward tracing (AUX): should finish on the DAC ###
        #################################################
        ((block,port),) = soc.metadata.trace_bus(self.fullpath, self.STREAM_OUT_AUX_PORT)

        while True:
            blocktype = soc.metadata.mod2type(block)

            if blocktype == "usp_rf_data_converter":
                self.HAS_AUX = True
                dac = port[1:3]
                self.cfg['dac_aux'] = dac
                self.cfg['fs_aux']  = soc.dacs[dac]['fs']/16
                break
            elif blocktype == "axis_terminator":
                break
            else:
                raise RuntimeError("falied to trace port for %s - unrecognized IP block %s" % (self.fullpath, block))

        # Configure block.
        self.configure()
        
    def sweep_config(self, config, debug = False, plot = False):
        # Check if channel is defined.
        if 'channel' not in config.keys():
            raise RuntimeError("%s: channel must be defined" % self.__class__.__name__)
        elif ( config['channel'] < 0 or config['channel'] > self.NDDS-1):
            raise RuntimeError("%s: channel must be in [0, %d]" % (self.__class__.__name__, self.NDDS-1))

        # Check if memory address is defined.
        if 'address' not in config.keys():
            raise RuntimeError("%s: address must be defined" % self.__class__.__name__)
        elif ( config['address'] < 0 or config['address'] > self.MEM_LENGTH):
            raise RuntimeError("%s: address must be in [0, %d]" % (self.__class__.__name__, self.MEM_LENGTH-1))

        # Check if freq_y is defined.
        if 'freq_y' not in config.keys():
            raise RuntimeError("%s: Frequency must be defined" % self.__class__.__name__)

        # Define order.
        config['freq_order'] = len(config['freq_y']) - 1

        # Check if freq_x is defined.
        if 'freq_x' not in config.keys():
            config['freq_x'] = np.linspace(0,1,len(config['freq_y']))

        # Check if freq_gain is defined.
        if 'freq_gain' not in config.keys():
            config['freq_gain'] = 0.99

        # Check if amp_y is defined.
        if 'amp_y' not in config.keys():
            raise RuntimeError("%s: Amplitude must be defined" % self.__class__.__name__)

        # Define order.
        config['amp_order'] = len(config['amp_y']) - 1

        # Check if amp_x is defined.
        if 'amp_x' not in config.keys():
            config['amp_x'] = np.linspace(0,1,len(config['amp_y']))

        # Check if amp_gain is defined.
        if 'amp_gain' not in config.keys():
            config['amp_gain'] = 0.99

        # Check if phase is defined.
        if 'phase' not in config.keys():
            config['phase'] = 0

        ############################
        ### Frequency Modulation ###
        ############################
        if (debug):
            print("############################")
            print("### Frequency Modulation ###")
            print("############################")

        # Check frequency vector.
        if np.max(np.abs(config['freq_y']) >= self.cfg['fs']/2):
            raise RuntimeError("%s: Frequency must be in [{%f},{%f}]" %(__class__.__name__, -self.cfg['fs']/2, self.cfg['fs']/2))

        # Check order.
        if ( (config['freq_order'] < self.FMOD_MIN_ORDER) or (config['freq_order'] > self.FMOD_MAX_ORDER)):
            raise RuntimeError("%s: Modulation order must be in [%d, %d]" %(__class__.__name__, self.FMOD_MIN_ORDER, self.FMOD_MAX_ORDER))

        # Input points.
        x = np.array(config['freq_x'])
        y = np.array(config['freq_y'])/(self.cfg['fs']/2)

        if (debug):
            for i in np.arange(len(x)):
                print("x[{}] = {:.3f}".format(i,x[i]))
            
            for i in np.arange(len(x)):
                print("y[{}] = {:.3f}".format(i,y[i]))            

        # Gain.
        g = config['freq_gain']

        # Polynomial fit.
        c = np.polyfit(x, y, config['freq_order'])
        p = np.poly1d(c)

        # Sort coefs from lower to higher degree.
        c = np.flip(c)
        c = np.concatenate((c,np.zeros(self.FMOD_MAX_ORDER-config['freq_order'])))

        # Coefficient Quantization.
        # IL: integer length. Include the sign bit.
        IL = int(np.ceil(np.log2(max(abs(c)))) + 1)
        if (IL < 1):
            IL = 1

        FL = int(self.BFREQ - IL)

        # Initialize regs structure.
        config['regs'] = {}

        # Write coefficients into regs structure.
        config['regs']['fmod_c0_reg'] = int(c[0]*(2**FL))
        config['regs']['fmod_c1_reg'] = int(c[1]*(2**FL))
        config['regs']['fmod_c2_reg'] = int(c[2]*(2**FL))
        config['regs']['fmod_c3_reg'] = int(c[3]*(2**FL))    
        config['regs']['fmod_c4_reg'] = int(c[4]*(2**FL))    
        config['regs']['fmod_c5_reg'] = int(c[5]*(2**FL))    

        if (debug):
            print("max(abs(c)) = {:.3f}, IL = {:d}, FL = {:d}".format(max(abs(c)),IL,FL))
            print("Coefficient format: Q{}.{}".format(IL,FL))

            for i in np.arange(len(c)):
                print("c[{}] = {:.3f}".format(i,c[i]))
                print("c_int[{}] = {}".format(i,int(np.round(c[i]*(2**FL)))))

            print("")

        # Gain Quantization.
        # I want product output to be Q6.X, where X is 2*B - 6.
        # The number of available IL is 6 - IL(coef).
        ILg = 6 - IL;
        FLg = self.BFREQ - ILg;

        # Write gain into regs structure.
        config['regs']['fmod_g_reg'] = int(g*(2**FLg))    

        if (debug):
            print("Gain format: Q{}.{}".format(ILg,FLg))
            print("g = {}".format(g))
            print("g_int = {}".format(int(np.round(g*(2**FLg)))))

        if (plot):
            xx = np.linspace(min(x), max(x), 100)
            yy = g*p(xx)

            plt.figure(dpi=100);
            plt.plot(xx*self.cfg['SWEEP_TIME_US'],self.cfg['fs']/2*yy,'-b',x*self.cfg['SWEEP_TIME_US'],self.cfg['fs']/2*g*y,'r*');
            plt.xlabel("t [us]");
            plt.ylabel("f [MHz]");
            plt.title("Frequency Modulation");

        ############################
        ### Amplitude Modulation ###
        ############################
        if (debug):
            print("############################")
            print("### Amplitude Modulation ###")
            print("############################")

        # Check amplitude vector.
        if np.max(np.abs(config['amp_y']) >= 1.0):
            raise RuntimeError("%s: Amplitude must be in [-1.0,1.0]" %(__class__.__name__))

        # Check order.
        if ( (config['amp_order'] < self.AMOD_MIN_ORDER) or (config['amp_order'] > self.AMOD_MAX_ORDER)):
            raise RuntimeError("%s: Modulation order must be in [%d, %d]" %(__class__.__name__, self.AMOD_MIN_ORDER, self.AMOD_MAX_ORDER))

        # Input points.
        x = np.array(config['amp_x'])
        y = np.array(config['amp_y'])

        if (debug):
            for i in np.arange(len(x)):
                print("x[{}] = {:.3f}".format(i,x[i]))
            
            for i in np.arange(len(x)):
                print("y[{}] = {:.3f}".format(i,y[i]))            

        # Gain.
        g = config['amp_gain']

        # Polynomial fit.
        c = np.polyfit(x, y, config['amp_order'])
        p = np.poly1d(c)

        # Sort coefs from lower to higher degree.
        c = np.flip(c)
        c = np.concatenate((c,np.zeros(self.AMOD_MAX_ORDER-config['amp_order'])))

        # Coefficient Quantization.
        # IL: integer length. Include the sign bit.
        IL = int(np.ceil(np.log2(max(abs(c)))) + 1)
        if (IL < 1):
            IL = 1

        FL = int(self.BAMP - IL)

        # Write coefficients into regs structure.
        config['regs']['amod_c0_reg'] = int(c[0]*(2**FL))
        config['regs']['amod_c1_reg'] = int(c[1]*(2**FL))
        config['regs']['amod_c2_reg'] = int(c[2]*(2**FL))
        config['regs']['amod_c3_reg'] = int(c[3]*(2**FL))    

        if (debug):
            print("max(abs(c)) = {:.3f}, IL = {:d}, FL = {:d}".format(max(abs(c)),IL,FL))
            print("Coefficient format: Q{}.{}".format(IL,FL))

            for i in np.arange(len(c)):
                print("c[{}] = {:.3f}".format(i,c[i]))
                print("c_int[{}] = {}".format(i,int(np.round(c[i]*(2**FL)))))

            print("")

        # Gain Quantization.
        # I want product output to be Q4.X, where X is 2*B - 4.
        # The number of available IL is 4 - IL(coef).
        ILg = 4 - IL;
        FLg = self.BAMP - ILg;

        # Write gain into regs structure.
        config['regs']['amod_g_reg'] = int(g*(2**FLg))    

        if (debug):
            print("Gain format: Q{}.{}".format(ILg,FLg))
            print("g = {}".format(g))
            print("g_int = {}".format(int(np.round(g*(2**FLg)))))

        if (plot):
            xx = np.linspace(min(x), max(x), 100)
            yy = g*p(xx)

            plt.figure(dpi=100);
            plt.plot(xx*self.cfg['SWEEP_TIME_US'],yy,'-b',x*self.cfg['SWEEP_TIME_US'],g*y,'r*');
            plt.xlabel("t [us]");
            plt.ylabel("A [Norm]");
            plt.title("Amplitude Modulation");                

        # Phase register.
        config['regs']['poff_reg'] = int(np.round(config['phase']/self.dphi))

        if (debug):
            print("phase = {}".format(config['phase']))
            print("poff_reg = {}".format(config['regs']['poff_reg']))

    def sweep_config_m(self, config, debug = False, plot = False):
        Ncfg = len(config)

        for i in np.arange(Ncfg):
            cfg = config[i]
            self.sweep_config(cfg, debug = debug, plot = plot)
        
    def sweep_config_write(self, config, debug = False):
        # Route switch to channel.
        self.switch.sel(mst=self.switch_ch)

        # Allocate buffer.
        buff = allocate(shape=(self.NREG+1,), dtype=np.int32)

        buff[0]  = config['channel']
        buff[1]  = config['address']
        buff[2]  = config['regs']['fmod_c0_reg']
        buff[3]  = config['regs']['fmod_c1_reg']
        buff[4]  = config['regs']['fmod_c2_reg']
        buff[5]  = config['regs']['fmod_c3_reg']
        buff[6]  = config['regs']['fmod_c4_reg']
        buff[7]  = config['regs']['fmod_c5_reg']
        buff[8]  = config['regs']['fmod_g_reg']
        buff[9]  = config['regs']['amod_c0_reg']
        buff[10] = config['regs']['amod_c1_reg']
        buff[11] = config['regs']['amod_c2_reg']
        buff[12] = config['regs']['amod_c3_reg']
        buff[13] = config['regs']['amod_g_reg']
        buff[14] = config['regs']['poff_reg']
        buff[15] = 0 # NOTE: control not used.

        if (debug):
            for i in np.arange(len(buff)):
                print("{}: buff[{}] = {}".format(__class__.__name__,i,buff[i]))

        # Get ready to write registers through DMA.
        self.memw_start_reg = 1

        # DMA data.
        self.dma.sendchannel.transfer(buff)
        self.dma.sendchannel.wait()

        # Stop write registers through DMA.
        self.memw_start_reg = 0

        # Free buffer.
        buff.freebuffer()

    def sweep_config_write_m(self, config, debug = False):
        # Route switch to channel.
        self.switch.sel(mst=self.switch_ch)

        # Number of configurations.
        Ncfg = len(config)

        # Allocate buffer.
        buff = allocate(shape=(Ncfg*(self.NREG+1),), dtype=np.int32)

        for i in np.arange(Ncfg):
            cfg = config[i]

            buff[i*(self.NREG+1) + 0]  = cfg['channel']
            buff[i*(self.NREG+1) + 1]  = cfg['address']
            buff[i*(self.NREG+1) + 2]  = cfg['regs']['fmod_c0_reg']
            buff[i*(self.NREG+1) + 3]  = cfg['regs']['fmod_c1_reg']
            buff[i*(self.NREG+1) + 4]  = cfg['regs']['fmod_c2_reg']
            buff[i*(self.NREG+1) + 5]  = cfg['regs']['fmod_c3_reg']
            buff[i*(self.NREG+1) + 6]  = cfg['regs']['fmod_c4_reg']
            buff[i*(self.NREG+1) + 7]  = cfg['regs']['fmod_c5_reg']
            buff[i*(self.NREG+1) + 8]  = cfg['regs']['fmod_g_reg']
            buff[i*(self.NREG+1) + 9]  = cfg['regs']['amod_c0_reg']
            buff[i*(self.NREG+1) + 10] = cfg['regs']['amod_c1_reg']
            buff[i*(self.NREG+1) + 11] = cfg['regs']['amod_c2_reg']
            buff[i*(self.NREG+1) + 12] = cfg['regs']['amod_c3_reg']
            buff[i*(self.NREG+1) + 13] = cfg['regs']['amod_g_reg']
            buff[i*(self.NREG+1) + 14] = cfg['regs']['poff_reg']
            buff[i*(self.NREG+1) + 15] = 0

        if (debug):
            for i in np.arange(len(buff)):
                print("{}: buff[{}] = {}".format(__class__.__name__,i,buff[i]))

        # Get ready to write registers through DMA.
        self.memw_start_reg = 1

        # DMA data.
        self.dma.sendchannel.transfer(buff)
        self.dma.sendchannel.wait()

        # Stop write registers through DMA.
        self.memw_start_reg = 0

        # Free buffer.
        buff.freebuffer()

    def set_single(self, ch = 0, addr = 0, f = 0, g = 0, debug = False):
        # This function won't use higher order coefficients for both frequency and amplitude modulation.
        #
        # Fixed Point:
        # fmod_c0_reg   : Q5.13
        # fmod_g_reg    : Q1.17
        # amod_c0_reg   : Q2.16
        # amod_g_reg    : Q1.17
        #
        # Both fmod_c0_reg and amod_c0_reg are set to 1.
        # fmod_g_reg and amod_g_reg are used to control final values.

        # Sanity checks.
        if ( ch >= self.NDDS ):
            print("{}: ch = {} must be less than {}".format(__class__.__name__, ch, self.NDDS))
            return

        if ( addr >= self.MEM_LENGTH):
            print("{}: addr = {} must be less than {}".format(__class__.__name__, addr, self.MEM_LENGTH))

        if ( np.abs(g) >= 1 ):
            print("{}: g = {} must be in (-1..1)".format(__class__.__name__, g))
            return

        if ( f >= self.cfg['fs']/2 ):
            print("{}: f must be in [0,{})".format(__class__.__name__, self.cfg['fs']/2))
            return

        addr_reg    = addr
        fmod_c0_reg = 2**13-1                           # 1.0 in Q5.13 format.
        fmod_c1_reg = 0
        fmod_c2_reg = 0
        fmod_c3_reg = 0
        fmod_c4_reg = 0
        fmod_c5_reg = 0
        fmod_g_reg  = int(round(f/self.cfg['df']))      # Q1.x format.
        amod_c0_reg = 2**11-1                           # 1.0 in Q5.11 format.
        amod_c1_reg = 0
        amod_c2_reg = 0
        amod_c3_reg = 0
        amod_g_reg  = int(round(g*(2**(self.BAMP-1))))  # Q1.x format.
        poff_reg    = 0
        ctrl_reg    = 0

        if (debug):
            print("addr_reg    = {}".format(addr_reg))
            print("fmod_c0_reg = {}".format(fmod_c0_reg))
            print("fmod_g_reg  = {}".format(fmod_g_reg))
            print("amod_c0_reg = {}".format(amod_c0_reg))
            print("amod_g_reg  = {}".format(amod_g_reg))

        # Route switch to channel.
        self.switch.sel(mst=self.switch_ch)

        # Allocate buffer.
        buff = allocate(shape=(self.NREG+1,), dtype=np.uint32)
        buff[0]  = ch
        buff[1]  = addr_reg
        buff[2]  = fmod_c0_reg
        buff[3]  = fmod_c1_reg
        buff[4]  = fmod_c2_reg
        buff[5]  = fmod_c3_reg
        buff[6]  = fmod_c4_reg
        buff[7]  = fmod_c5_reg
        buff[8]  = fmod_g_reg
        buff[9]  = amod_c0_reg
        buff[10] = amod_c1_reg
        buff[11] = amod_c2_reg
        buff[12] = amod_c3_reg
        buff[13] = amod_g_reg
        buff[14] = poff_reg
        buff[15] = ctrl_reg

        # Get ready to write memory through DMA.
        self.memw_start_reg = 1

        # DMA data.
        self.dma.sendchannel.transfer(buff)
        self.dma.sendchannel.wait()

        # Stop write registers through DMA.
        self.memw_start_reg = 0

        # Free buffer.
        buff.freebuffer()
        
    def alloff(self, addr = 0, debug = False):
        for i in np.arange(self.NDDS):
            if (debug):
                print("{}: setting ch  = {}, addr = {}".format(__class__.__name__, i, addr))

            self.set_single(ch = i, addr = addr, debug = debug)

    def alloff_mem(self, debug = False):
        for i in np.arange(self.MEM_LENGTH):
            self.alloff(addr = i)

class QickAmoSoc(QickSoc):
    def __init__(self, bitfile=None, force_init_clks=False,ignore_version=True, **kwargs):
        QickSoc.__init__(self, bitfile=bitfile, force_init_clks=force_init_clks, ignore_version=ignore_version, **kwargs)
        
    # Extend map_signal_paths.
    def map_signal_paths(self):
        # Run standard QickSoc map_signal_paths.        
        super().map_signal_paths()
        
        # Add Signal Generators AMO V3 to gen list.
        for key, val in self.ip_dict.items():
            if val['driver'] == AxisSignalGeneratorAMOV4:
                self.gens.append(getattr(self,key))

    # Replace description to add AMO generators.
    def description(self):
        tproc = self['tprocs'][0]
        
        lines = []
        lines.append("\n\tBoard: " + self['board'])
        lines.append("\n\tSoftware version: " + self['sw_version'])
        lines.append("\tFirmware timestamp: " + self['fw_timestamp'])
        lines.append("\n\tGlobal clocks (MHz): tProcessor %.3f, RF reference %.3f" % (
            tproc['f_time'], self['refclk_freq']))
        
        # Signal Generators.
        lines.append("\n\t%d signal generator channels:" % (len(self.gens)))
        for iGen, gen in enumerate(self.gens):
            if gen['type'] == "axis_signal_gen_amo_v4":
                lines.append("\t%d:\t%s - tProc output %d, NDDS = %d, parameter memory %d" %(iGen,gen['type'], gen['tproc_ch'], gen.NDDS, gen.MEM_LENGTH))
                lines.append("\t\tDAC tile %d, blk %d, fs = %.3f MHz, df = %.3f kHz, min sweep = %.3f us" % (int(gen['dac'][0]), int(gen['dac'][1]), gen['fs'], 1000*gen.cfg['df'],gen.cfg['SWEEP_TIME_US']))
                if gen.HAS_AUX:
                    lines.append("\t\tDAC tile %d, blk %d: Auxiliary Output" % (int(gen['dac_aux'][0]), int(gen['dac_aux'][1])))
            else:
                lines.append("\t%d:\t%s - tProc output %d, envelope memory %d samples" %
                             (iGen, gen['type'], gen['tproc_ch'], gen['maxlen']))
                lines.append("\t\tDAC tile %s, blk %s, %d-bit DDS, fabric=%.3f MHz, f_dds=%.3f MHz" %
                             (*gen['dac'], gen['b_dds'], gen['f_fabric'], gen['f_dds']))
        
        lines.append("\n\t%d DACs:" % (len(self['dacs'])))
        for dac in self['dacs']:
            tile, block = [int(c) for c in dac]
            if self['board']=='ZCU111':
                label = "DAC%d_T%d_CH%d or RF board output %d" % (tile + 228, tile, block, tile*4 + block)
            elif self['board']=='ZCU216':
                label = "%d_%d, on JHC%d" % (block, tile + 228, 1 + (block%2) + 2*(tile//2))
            elif self['board']=='RFSoC4x2':
                label = {'00': 'DAC_B', '20': 'DAC_A'}[dac]
            lines.append("\t\tDAC tile %d, blk %d is %s" %
                         (tile, block, label))
        
        lines.append("\n\t%d ADCs:" % (len(self['adcs'])))
        for adc in self['adcs']:
            tile, block = [int(c) for c in adc]
            if self['board']=='ZCU111':
                rfbtype = "DC" if tile > 1 else "AC"
                label = "ADC%d_T%d_CH%d or RF board %s input %d" % (tile + 224, tile, block, rfbtype, (tile%2)*2 + block)
            elif self['board']=='ZCU216':
                label = "%d_%d, on JHC%d" % (block, tile + 224, 5 + (block%2) + 2*(tile//2))
            elif self['board']=='RFSoC4x2':
                label = {'00': 'ADC_D', '01': 'ADC_C', '20': 'ADC_B', '21': 'ADC_A'}[adc]
            lines.append("\t\tADC tile %d, blk %d is %s" %
                         (tile, block, label))
        
        lines.append("\n\t%d digital output pins:" % (len(tproc['output_pins'])))
        for iPin, (porttype, port, pin, name) in enumerate(tproc['output_pins']):
            lines.append("\t%d:\t%s (%s %d, pin %d)" % (iPin, name, porttype, port, pin))
        
        lines.append("\n\ttProc %s: program memory %d words, data memory %d words" %
                (tproc['type'], tproc['pmem_size'], tproc['dmem_size']))
        lines.append("\t\texternal start pin: %s" % (tproc['start_pin']))
        
        return "\nQICK configuration:\n"+"\n".join(lines)

