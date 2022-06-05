// task list:
// One) Trade onBar on from indicator
// check indicator if indicator and position match, do nothing.  
// If indicator does not match position: trade 2x volume in indicated direction.
// Result should be closing of current position and oppening of new.  



using System;
using System.Linq;
using cAlgo.API;
using cAlgo.API.Indicators;
using cAlgo.API.Internals;
using cAlgo.Indicators;
using System.IO;
using System.Windows.Forms;
using System.Text;
using System.Diagnostics;
using System.Threading.Tasks;
namespace cAlgo
{
    [Robot(TimeZone = TimeZones.UTC, AccessRights = AccessRights.FullAccess)]
    public class ExportAllData : Robot
    {

        //[Parameter(DefaultValue = 0.0)]
        //public double Parameter { get; set; }
        //[Parameter(Limit = 0.0025)]


        protected override void OnStart()
        {

            //exportInitialData(constants.fileNameData, constants.fileNameTime);




        }

        protected override void OnBar()
        {



            exportCurrentData(constants.fileNameData, constants.fileNameTime, constants.fileNameDataHigh, constants.fileNameDataLow);

            //analyze(constants.fileNameData, constants.fileNameAnalyze);







        }


        protected override void OnTick()
        {
            //exportCurrentTicData(constants.fileNameData, constants.fileNameTime);

        }

        protected override void OnStop()
        {

            // Put your deinitialization logic here
        }




















        ///////////////////////////////////////////////////////////////////////////////////////////////
        //Functions here:
        //////////////////////////////////////////////////////////////////////////////////////////////
        public class constants
        {
            //public const string sym = cu;
            public const string workingDirectory = "C:\\Users\\Derre\\Desktop\\data\\scripts";
            public const string fileNameData = "dataOut.txt";
            public const string fileNameDataHigh = "dataOut_High.txt";
            public const string fileNameDataLow = "dataOut_Low.txt";
            public const string fileNameTime = "timeStamp.txt";
            public const string fileNameAnalyze = "analyze.py";
            
            //public const double limit = 0.001;
            //public const double stop = 0.001;
            public const double LFratio = 0.001;
            public const string fileNameIndicator = "indicator.txt";
            public const int listlength = 4000;
            //public const long vol = 10000;
        }


       

        public void exportCurrentData(string fileNameData, string fileNameTime, string fileNameDataHigh, string fileNameDataLow)
        {
            var symbolName = Bars.SymbolName;
            
            var desktopFolder = constants.workingDirectory;
            var filePathData = Path.Combine(desktopFolder, symbolName + "_1m_" + fileNameData);
            var filePathTime = Path.Combine(desktopFolder, symbolName + "_1m_" + fileNameTime);
            var filePathDataHigh = Path.Combine(desktopFolder, fileNameDataHigh);
            var filePathDataLow = Path.Combine(desktopFolder, fileNameDataLow);

            int listlength = constants.listlength;

            //double last = new double;
            //string lastStr = new string[listlength];
            //DateTime tictime = new DateTime[listlength];
            //string tictimeStr = new string[listlength];

            string last =  Bars.OpenPrices.Last(0).ToString("F6");// MarketSeries.Close.Last(0).ToString("F6");
            //string last_high =  Bars.HighPrices.Last(0).ToString("F6");// MarketSeries.Close.Last(0).ToString("F6");
            //string last_low = Bars.LowPrices.Last(0).ToString("F6");// MarketSeries.Close.Last(0).ToString("F6");
            Print(last);
            string tictime = Bars.OpenTimes.Last(0).ToString("yyyy-MM-dd H:mm:ss.fff"); //MarketSeries.OpenTime.Last(0).ToString("yyyy-MM-dd H:mm:ss");
            //Print(Bars.OpenPrices.Last(0));
            
            File.AppendAllText(filePathData, last + Environment.NewLine);
            //File.AppendAllText(filePathDataHigh, last_high + Environment.NewLine);
            //File.AppendAllText(filePathDataLow, last_low + Environment.NewLine);
            File.AppendAllText(filePathTime, tictime + Environment.NewLine);
            
        }




       
    }



}



