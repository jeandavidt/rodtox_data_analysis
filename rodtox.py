import pandas as pd
import numpy as np
import plotly as py
import plotly.graph_objs as go
import plotly.offline as offline
import ipywidgets as widgets
from scipy import special
import scipy.stats
import scipy.optimize
import scipy.integrate

def Plotit(df, Starttime, Endtime, List_y1, List_y2, Label1, Label2, Units1, Units2, marks):
    dashes = ['solid', 'dot', 'dashdot', 'solid', 'dot', 'dashdot']

    symbollist = [22, 23, 19, 20, 4, 17, 1]
    colorlist_1 = ['#5385A1']
    colorlist_2 = ['#70CC87', '#3E1D1E']
    colorlist_3 = ['#70CC87', '#546D8A', '#3E1D1E']
    colorlist_4 = ['#70CC87', '#3A909D', '#5F4C68', '#3E1D1E']
    colorlist_5 = ['#70CC87', '#3A909D', '#546D8A', '#5F4C68', '#3E1D1E']
    colorlist_6 = ['#70CC87', '#34AA9D', '#458298', '#5D5977', '#5A3648', '#3E1D1E']
    colorlist_7 = ['#70CC87', '#39B09A', '#3A909D', '#546D8A', '#5F4C68', '#563140', '#3E1D1E']
    colorlist_8 = ['#70CC87', '#3FB499', '#34999E', '#4A7C95', '#5B5F7D', '#5F445C', '#542D3B', '#3E1D1E']

    markerlist = []
    Number_of_lines = len(List_y1) + len(List_y2)
    for i in range(Number_of_lines):
        if Number_of_lines == 1:
            dictA = {'color': colorlist_1[i], 'size': 10, 'symbol': symbollist[i]}
            markerlist.append(dictA)
        elif Number_of_lines == 2:
            dictA = {'color': colorlist_2[i], 'size': 10, 'symbol': symbollist[i]}
            markerlist.append(dictA)
        elif Number_of_lines == 3:
            dictA = {'color': colorlist_3[i], 'size': 10, 'symbol': symbollist[i]}
            markerlist.append(dictA)
        elif Number_of_lines == 4:
            dictA = {'color': colorlist_4[i], 'size': 10, 'symbol': symbollist[i]}
            markerlist.append(dictA)
        elif Number_of_lines == 5:
            dictA = {'color': colorlist_5[i], 'size': 10, 'symbol': symbollist[i]}
            markerlist.append(dictA)
        elif Number_of_lines == 6:
            dictA = {'color': colorlist_6[i], 'size': 10, 'symbol': symbollist[i]}
            markerlist.append(dictA)
        elif Number_of_lines == 7:
            dictA = {'color': colorlist_7[i], 'size': 10, 'symbol': symbollist[i]}
            markerlist.append(dictA)
        elif Number_of_lines == 8:
            dictA = {'color': colorlist_8[i], 'size': 10, 'symbol': symbollist[i]}
            markerlist.append(dictA)

    x = df[Starttime:Endtime].index
    # x = pd.to_timedelta(x-x[0]).total_seconds()
    # x=x/3600

    Lab1 = ''
    for i in range(len(Label1)):
        if len(Label1) == 1:
            Lab1 += Label1[i] + ' ({}) '.format(Units1[i])
        elif i == len(Label1) - 1:
            Lab1 += 'and ' + Label1[i] + ' ({}) '.format(Units1[i])
        else:
            Lab1 += Label1[i] + ' ({}), '.format(Units1[i])

    Lab2 = ''
    for i in range(len(Label2)):
        if len(Label2) == 1:
            Lab2 += Label2[i] + ' ({}) '.format(Units2[i])
        elif i == len(Label2) - 1:
            Lab2 += 'and ' + Label2[i] + ' ({}) '.format(Units2[i])
        else:
            Lab2 += Label2[i] + ' ({}), '.format(Units2[i])
    layout = go.Layout(
        font=dict(
            family='PT Sans Narrow',
            size=20,
            color='#000000'
        ),
        yaxis=dict(
            title=Lab1,
            ticks='outside',
            titlefont=dict(
                color='#000000',
                size=20
            ),
            autorange=True,
            rangemode='tozero',  # normal or nonnegative
            linecolor='#000000',
            zeroline=False,
            showline=True,
        ),
        xaxis=dict(
            showticklabels=True,
            showline=True,
            ticks='outside',
            title='Time (h)',
            titlefont=dict(
                color='#000000',
                size=20
            ),
            # tickangle=45,
            # tickformat='%m/%d %H:%M'
        ),

        legend=dict(
            x=-0.35,
            y=1,
            traceorder='normal',
            font=dict(
                family='PT Sans Narrow',
                size=20,
                color='#000000'
            ),
        ),
        autosize=False,
        width=900,
        height=500,
        margin=go.layout.Margin(
            l=50,
            r=100,
            b=100,
            t=50,
            pad=4
        ),
        paper_bgcolor='#FFFFFF',
        plot_bgcolor='#FFFFFF'

    )
    yaxis2 = {
        'yaxis2': {
            'title': Lab2,
            'ticks': 'outside',
            'titlefont': {'color': '#000000', 'size': 20},
            'autorange': True,
            'linecolor': '#000000',
            'zeroline': False,
            'showline': True,
            'anchor': 'x',
            'overlaying': 'y',
            'side': 'right',
            'rangemode': 'tozero',
            'showgrid': False
        }
    }
    if len(List_y2) != 0:
        layout.update(yaxis2)

    data = []
    for i in range(len(List_y1)):
        trace1 = go.Scattergl(
            x=x,
            y=df.loc[Starttime:Endtime, List_y1[i]],
            mode=marks[0],  # markers, lines, or lines+markers
            marker=markerlist[i],
            name='{}'.format(Label1[i]),
            yaxis='y1',
            line=dict(
                dash=dashes[i])
        )
        data.append(trace1)

    # series2 = ['stBOD']
    for i in range(len(List_y2)):
        trace2 = go.Scattergl(
            x=x,
            y=df.loc[Starttime:Endtime, List_y2[i]],
            mode=marks[1],  # markers, lines, or lines+markers
            marker=markerlist[i + len(List_y1)],
            name='{}'.format(Label2[i]),
            yaxis='y2',
            line=dict(
                dash=dashes[i + len(List_y1)])
        )
        data.append(trace2)

    fig = go.Figure(data=data, layout=layout)

    return fig

def ImportRdtxCsv(csvDO, csvTemp, sourcepath, destinationpath, existing=None):
    # Create the complete filepath to get to the data
    filepath = sourcepath + '/' + csvDO
    # Read the.csv file and turn it into a DataFrame
    RawDO = pd.read_csv(filepath, sep=';')
    # Remove invalid entries from the data
    RawDO = RawDO[RawDO.Validity == 1]
    # Only keep rows where the probe sends a DO measurement
    RawDO = RawDO[RawDO.VarName == 'HMI_DO']
    # Replace commas by dots so that DO values are treated as floats instead of strings
    RawDO.replace({',': '.'}, regex=True, inplace=True)
    # Transform the timestamps into machine-readable DateTime objects
    RawDO['Time'] = pd.to_datetime(RawDO.TimeString, format='%d.%m.%Y %H:%M:%S')
    # Assign new name to the columns
    RawDO.columns = ['VarName', 'TimeString', 'DO', 'Validity', 'Time_ms', 'Time']
    # Set the 'Time' column as the DataFrame's index
    RawDO.set_index(RawDO.Time, inplace=True)  # set the Time column as the DataFrame's index
    # Transform the DO values from strings into numbers
    RawDO.DO = pd.to_numeric(RawDO.DO)

    # Repeat the same process for Temperature data.
    filepath = sourcepath + '/' + csvTemp
    RawTemp = pd.read_csv(filepath, sep=';')
    RawTemp = RawTemp[RawTemp.Validity == 1]
    RawTemp = RawTemp[RawTemp.VarName == 'HMI_Temp']
    RawTemp.replace({',': '.'}, regex=True, inplace=True)
    RawTemp['Time'] = pd.to_datetime(RawTemp.TimeString, format='%d.%m.%Y %H:%M:%S')
    RawTemp.columns = ['VarName', 'TimeString', 'Temp', 'Validity', 'Time_ms', 'Time']
    RawTemp.set_index(RawTemp.Time, inplace=True)
    RawTemp.Temp = pd.to_numeric(RawTemp.Temp)

    # Concatenate the temperature and dissolved oxygen data, and drop useless columns and empty rows
    df = pd.concat([RawDO.DO, RawTemp.Temp], axis=1, keys=['DO', 'Temp'])
    df.dropna(inplace=True)
    df.sort_index(inplace=True)

    # If we do not want to merge the new data with existing files, save the new data to a new file
    if existing is None:
        # Find the earliest timestamp
        Start = str(df.first_valid_index())
        # Find the latest timestamp
        End = str(df.last_valid_index())
        # Save the resulting file to the correct directory
        dest_filepath = destinationpath + '/' + str(Start)[:10] + '_' + str(End)[:10] + '.csv'
        df.to_csv(dest_filepath, sep=';')

    # If we do want to merge the new data with an existing file:
    else:
        # read the existing file
        filepath = destinationpath + '/' + existing
        df_ex = pd.read_csv(filepath, sep=';')
        df_ex.Time = pd.to_datetime(df_ex.Time)
        df_ex.DO = pd.to_numeric(df_ex.DO)
        df_ex.Temp = pd.to_numeric(df_ex.Temp)
        df_ex.set_index(df_ex.Time, drop=True, inplace=True)

        # Append one data frame to another
        df_ex = df_ex.append(df)
        df_ex.drop_duplicates(inplace=True)
        df_ex.sort_index(inplace=True)

        # Find the earliest timestamp
        Start = str(df_ex.first_valid_index())
        # Find the latest timestamp
        End = str(df_ex.last_valid_index())
        # Save the resulting file to the correct directory
        dest_filepath = destinationpath + '/' + str(Start)[:10] + '_' + str(End)[:10] + '.csv'
        df_ex.to_csv(dest_filepath, sep=';')
        df = df_ex
    return df, Start, End

def rolling_window(a, window):
    a = a.to_numpy()
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)

def Deriv(df, tau, n):
    # The rolling_window function helps to calculate moving-window statistics. © 2016 Erik Rigtorp
    # http://www.rigtorp.se/2011/01/01/rolling-statistics-numpy.html
    DF = df.copy(deep=True)
    if n % 2 == 0:
        n += 1
    NewColumns = ['DOdt_A', 'DOdt_smooth', 'DO_A', 'DO_smooth', 'seconds']
    for name in NewColumns:
        if name not in DF.columns:
            DF.insert(len(DF.columns), name, np.nan)
    DF.reset_index(inplace=True)
    DF['seconds'] = (DF['Time'] - DF['Time'].iloc[0]).dt.total_seconds()
    dist = int((n - 1) / 2)
    # Calculate the Derivative of the raw signal
    x = DF['seconds']
    y = DF['DO']

    RolledX = rolling_window(x, n)
    RolledY = rolling_window(y, n)
    x_mean = np.mean(RolledX, axis=1)
    y_mean = np.mean(RolledY, axis=1)
    x_xmean = np.transpose(np.transpose(RolledX) - x_mean)
    y_ymean = np.transpose(np.transpose(RolledY) - y_mean)

    prodxy = np.multiply(x_xmean, y_ymean)
    prodxx = np.multiply(x_xmean, x_xmean)

    Sumxy = np.sum(prodxy, axis=1)
    Sumxx = np.sum(prodxx, axis=1)
    Slope = np.divide(Sumxy, Sumxx)
    Intercept = y_mean - Slope * x_mean
    y_hat = Intercept + Slope * x[dist:-(dist)]

    # Assign the results to the right columns
    DF = DF.iloc[dist:-(dist)]
    DF['DOdt_smooth'] = Slope * 3600
    DF['DO_smooth'] = y_hat
    DF['DO_A'] = DF['DO_smooth'] + tau * DF['DOdt_smooth'] / 3600

    # Calculate the derivative of the DO_Actual time series
    x = DF['seconds']
    RolledX = rolling_window(x, n)
    x_mean = np.mean(RolledX, axis=1)
    x_xmean = np.transpose(np.transpose(RolledX) - x_mean)
    prodxx = np.multiply(x_xmean, x_xmean)
    Sumxx = np.sum(prodxx, axis=1)

    DOA = DF['DO_A']
    RolledDOA = rolling_window(DOA, n)
    DOA_mean = np.mean(RolledDOA, axis=1)
    DOA_DOAmean = np.transpose(np.transpose(RolledDOA) - DOA_mean)
    prodxDOA = np.multiply(x_xmean, DOA_DOAmean)
    SumxDOA = np.sum(prodxDOA, axis=1)
    SlopeDOA = np.divide(SumxDOA, Sumxx)
    DF = DF.iloc[dist:-(dist)]
    DF['DOdt_A'] = SlopeDOA * 3600

    DF.set_index(DF['Time'], inplace=True)
    DF = DF.drop(['Time'], 1)
    return DF

def calc_kla(df, Start, End, Sample_Volume, Cal_Volume):  # Volumes in liters
    # Make a copy of the DataFrame to work on #####
    DF = df[Start:End].copy(deep=True)

    # Add the new columns we need to the DataFrame #####
    NewColumns = [
        'Co', 'Ce', 'Kla', 'Area', 'stBOD', 'OURex', 'PeakNo', 'ReacVol',
        'DO_Start', 'DO_End', 'DO_Bend', 'DO_Bottom', 'x_plot', 'y_plot',
    ]
    for name in NewColumns:
        if name not in DF.columns:
            DF.insert(len(DF.columns), name, np.nan)

    # Find important points in peaks (Start, End, Bend, Bottom) #####
    # Filter the DataFrame to find the starting point of a respirogram
    Percentile = np.percentile(DF['DO_A'], 70)
    # Filtered= DF.loc[(DF.index==DF.first_valid_index()) | ((DF['DO_A'] > Percentile) & (DF['DOdt_A'].diff(-1) > 0) & (DF['DOdt_A'] > 0) & (DF['DOdt_A'].shift(-1) < 0)& (DF['DO_A']>DF['DO_A'].shift(-1)))]
    Filtered = DF.loc[(DF.index == DF.first_valid_index()) | ((DF['DO_A'] > Percentile) & (DF['DOdt_A'] > 0) & (DF['DOdt_A'].shift(-1) < 0))]

    Filtered['Start'] = pd.to_datetime(Filtered.index)
    Filtered['DO_Start'] = Filtered['DO_A']
    # remove false positives by removing the Start points that are too close together in time
    i = 0
    while i < len(Filtered) - 1:
        duration = pd.to_timedelta(Filtered['Start'].iloc[i + 1] - Filtered['Start'].iloc[i]).total_seconds()
        if duration <= pd.to_timedelta('5 minutes').total_seconds():
            Filtered = Filtered.drop(Filtered.index[i])
            i = 0
        else:
            i += 1
    # Add columns to the filtered DataFrame for the other key points in the respirogram
    Filtered = Filtered.reset_index(drop=True)

    Columns = ['End', 'Bend', 'Bottom']
    for name in Columns:
        Filtered.insert(len(Filtered.columns), name, np.nan)
    for i in range(len(Filtered) - 1):
        # Defining the end of each respirogram
        Filtered['End'].iloc[i] = Filtered['Start'].iloc[i + 1]
        Filtered['DO_End'].iloc[i] = DF['DO_A'][Filtered['End'].iloc[i]]

        S = Filtered['Start'].iloc[i]
        E = Filtered['End'].iloc[i]
        # Find the lowest point of each respirogram
        BottomDO = DF[S:E]['DO_A'].min()
        Bottom_pos = pd.to_datetime(DF[S:E].loc[DF[S:E]['DO_A'] == DF[S:E]['DO_A'].min()].index)

        Filtered['Bottom'].iloc[i] = Bottom_pos[0]
        Filtered['DO_Bottom'].iloc[i] = BottomDO
        # Finding the "Bend" in each respirogram - the point at which biodegradation is complete
        # The Bend corresponds to the last peak of the derivative over the span of the respirogram
        bendS = Filtered['Bottom'].iloc[i]
        bendE = Filtered['End'].iloc[i]
        benddf = DF[bendS:bendE].copy(deep=True)
        benddf.reset_index(inplace=True)
        n = 51
        dist = int((n - 1) / 2)
        x = benddf['Time'].iloc[dist:-dist]
        if len(benddf) <= 1.5 * n:
            Filtered['Bend'].iloc[i] = bendS
            Filtered['DO_Bend'].iloc[i] = np.nan
        else:
            RolledDO = rolling_window(benddf['DOdt_A'], n)
            RolledMax = np.max(RolledDO, axis=1)

            Compa = pd.DataFrame(
                {'DOdt': benddf['DOdt_A'].iloc[dist:-dist],
                    'Max': RolledMax,
                    'Time': x}
            )

            Compa = Compa.loc[Compa['DOdt'] == Compa['Max']]

            Compa.reset_index(inplace=True, drop=True)
            if len(Compa) == 0:
                Filtered['Bend'].iloc[i] = bendS
                Filtered['DO_Bend'].iloc[i] = np.nan
            else:
                LastMax = Compa.iloc[Compa.last_valid_index()]
                Filtered['Bend'].iloc[i] = LastMax['Time']
                Filtered['DO_Bend'].iloc[i] = DF['DO_A'].loc[Filtered['Bend'].iloc[i]]

        # If the interval between two respirograms is too large, the End of the first respirogram is set (arbitrarily) to 15 minutes after the end of biodegradation.
        if pd.to_timedelta(Filtered['End'].iloc[i] - Filtered['Start'].iloc[i]) > pd.to_timedelta('2 hours'):
            Set = pd.to_datetime(Filtered['Bend'].iloc[i]) + pd.to_timedelta('15 minutes')
            Filtered['End'].iloc[i] = pd.to_datetime(DF.iloc[[DF.index.get_loc(Set, method='nearest')]].index)[0]
            Filtered['DO_End'].iloc[i] = DF['DO_A'][Filtered['End'].iloc[i]]

    S = DF.first_valid_index()
    E = Filtered['Bottom'].iloc[0]

    TopDO = DF[S:E]['DO_A'].max()
    Top_pos = pd.to_datetime(DF[S:E].loc[DF[S:E]['DO_A'] == TopDO].index)
    Filtered['Start'].iloc[0] = Top_pos[0]
    Filtered['DO_Start'].iloc[0] = TopDO
    # Drops the last row of the filtered DataFrame, which only contains a Start point with no corresponding End, Bottom or Bend
    Filtered = Filtered[:-1]

    # Assign the Important points to the appropriate rows in the original DataFrame
    for i in range(len(Filtered)):
        DF['DO_Start'][Filtered['Start'][i]] = Filtered['DO_Start'][i]
        DF['DO_End'][Filtered['End'][i]] = Filtered['DO_End'][i]
        DF['DO_Bend'][Filtered['Bend'][i]] = Filtered['DO_Bend'][i]
        DF['DO_Bottom'][Filtered['Bottom'][i]] = Filtered['DO_Bottom'][i]

    # Plotting the Important points ######
    Label1 = ['DO']
    Label2 = ['Start', 'End', 'Bend', 'Bottom']
    Units1 = ['mg/l']
    Units2 = ['mg/l', 'mg/l', 'mg/l', 'mg/l']
    marks = ['lines', 'markers']
    start_plot = DF.first_valid_index()
    end_plot = DF.last_valid_index()
    figure = Plotit(DF, start_plot, end_plot, ['DO_A'], ['DO_Start', 'DO_End', 'DO_Bend', 'DO_Bottom'], Label1, Label2, Units1, Units2, marks)
    py.offline.iplot(figure)
    # Kla estimation ####

    # Define the re-aeration function under conditions without biodegradation

    def f(t, Kla, Ce, Co):
        return Ce - (Ce - Co) * np.exp(-Kla * (t))

    # Add relevant columns to the Filtered DataFrame
    Kla_columns = ['Co', 'Ce', 'Kla', 'std_err']
    for name in Kla_columns:
        if name not in Filtered.columns:
            Filtered.insert(len(Filtered.columns), name, np.nan)

    for i in range(len(Filtered)):
        if ((Filtered['Bend'].iloc[i] == Filtered['End'].iloc[i]) | (pd.to_timedelta(Filtered['End'].iloc[i] - Filtered['Bend'].iloc[i]).total_seconds() <= pd.to_timedelta('8 minutes').total_seconds())):
            print('Peak {} of {} is skipped: Interval too short.'.format(i + 1, len(Filtered)))
            continue
        else:
            # Define a DataFrame containing only the re-aeration phase of respirogram i
            T00 = Filtered['Bend'][i]
            Tf = Filtered['End'][i]
            print('Peak {} of {}. Start is at {}'.format(i + 1, len(Filtered), T00))

            Timedel = pd.to_timedelta(Tf - T00)
            rollingdf = DF[T00: T00 + 0.9 * Timedel]
            rollingdf['tvalue'] = rollingdf.index

            # Create a DataFrame to contain the results of non-linear regressions operated on sub-sections of the re-aeration phase
            SubResult = pd.DataFrame(index=range(0, int(0.5 * len(rollingdf))), columns=['Start', 'End', 'Ce', 'Co', 'Kla', 'std_err'])

            # Try to perform a non-linear regression on several sub-section of the re-aeration phases
            for j in range(len(rollingdf)):
                # The beginning of the range of the non-linear regression shifts one data point forward at each iteration
                # The final value in the range stays constant
                T0 = rollingdf['tvalue'].iloc[[j]][0]
                y_given = DF['DO_A'][T0:Tf].copy()
                y_frame = y_given.to_frame()
                y_frame.reset_index(drop=True, inplace=True)
                x_given = (DF[T0:Tf].index - T0).total_seconds()

                try:
                    # Define the bounds of the non-linear regression for parameters Kla, Ce, Co
                    param_bounds = ([0, 0, -20], [100 / 3600, 20, 20])
                    # Try to perform the non-linear regression
                    params, cov = scipy.optimize.curve_fit(f, x_given, y_frame['DO_A'], bounds=param_bounds)
                # Move on to the next sub-section if the non-linear regression fails
                except RuntimeError:
                    continue
                # If it doesn't fail, assign the results of the non-linear regression to the SubResult DataFrame
                else:
                    SubResult['Start'][j] = pd.to_datetime(T0)
                    SubResult['End'][j] = pd.to_datetime(Tf)
                    SubResult['Ce'][j] = params[1]
                    SubResult['Co'][j] = params[2]
                    SubResult['Kla'][j] = params[0]
                    perr = np.sqrt(np.diag(cov))
                    sterr = perr[0]
                    SubResult['std_err'][j] = sterr
                # And delete stored variables which are no longer needed
                finally:
                    del y_given
                    del y_frame
                    del x_given
            # If the non-linear regressions have succeeded at leat once over all the sub-sections of the re-aeration phase:
            if len(SubResult) != 0:
                # Select the values of Ce, Co and Kla for the iteration with the lowest standard error
                ''' TopDO = DF[S:E]['DO_A'].max()
                    Top_pos = pd.to_datetime(DF[S:E].loc[DF[S:E]['DO_A']== TopDO].index)
                    Filtered['Start'].iloc[0] = Top_pos[0]
                    Filtered['DO_Start'].iloc[0] = TopDO'''
                SubResult['std_err'] = SubResult['std_err']**2
                Min_err = SubResult['std_err'].min()
                index_position = SubResult.loc[SubResult['std_err'] == Min_err].index
                Filtered['Ce'][i] = SubResult['Ce'][index_position]
                Filtered['Co'][i] = SubResult['Co'][index_position]
                Filtered['Kla'][i] = SubResult['Kla'][index_position] * 3600
                Filtered['std_err'][i] = SubResult['std_err'][index_position]
                # Calculate the corresponding DO saturation concentration
            # If all non-linear regressions failed, move on to the next respirogram without assigning values
            else:
                continue
    # Fill the Filtered DataFrame with values from the nearest successful non-linear regression
    Columns = ['Ce', 'Co', 'Kla', 'std_err']
    for name in Columns:
        Filtered[name] = Filtered[name].fillna(method='ffill')
    # Assign the obtained Kla, Ce, Co and Cs values to the original DataFrame
    Columns = ['Kla', 'Ce', 'Co']
    for name in Columns:
        for i in range(len(Filtered)):
            DF[name][Filtered['Start'][i]:Filtered['End'][i]] = Filtered[name][i]

    # Re-calculate the expected DO concentrations during re-aeration using the found Kla, Ce and Co values and plot them
    def f2(t, Ce, Co, Kla, to):
        return Ce - (Ce - Co) * np.exp(-Kla * pd.to_timedelta(t - to).astype('timedelta64[s]'))

    DF['x_plot'] = pd.to_datetime(DF.index)
    for i in range(len(Filtered)):
        DF['y_plot'].loc[Filtered['Bend'][i]:Filtered['End'][i]] = f2(DF['x_plot'], DF['Ce'], DF['Co'], DF['Kla'] / 3600, Filtered['Bend'][i])

    Label1 = ['DO exp.', 'DO calc.', 'Ce']
    Label2 = ['KLa']
    Units1 = ['mg/l', 'mg/l', 'mg/l']
    Units2 = ['h-1']
    marks = ['lines', 'lines']
    start_plot = DF.first_valid_index()
    end_plot = DF.last_valid_index()
    figure = Plotit(DF, start_plot, end_plot, ['DO_A', 'y_plot', 'Ce'], ['Kla'], Label1, Label2, Units1, Units2, marks)
    py.offline.iplot(figure)
    return DF, Filtered

def calc_stbod(df, Filtered, Start, End, Sample_Volume, Cal_Volume, dec_time):
    # #### Make a copy of the DataFrame to work on #####
    DF = df[Start:End].copy(deep=True)
    # #### Identifying Decantation peaks and assigning Peak ID's ##############################
    for i in range(len(Filtered)):
        # Check if a respirogram's DO falls below the threshold under which only decantation cycles fall
        # if Filtered['DO_Bottom'].iloc[i] <= float(Decthresh):
        Timedelta = (Filtered['Bottom'].iloc[i] - Filtered['Start'].iloc[i]).total_seconds()
        if (Timedelta >= dec_time):
            # Reset the Peak identification number and the reactor volume
            Filtered['PeakNo'].iloc[i] = 0
            Filtered['ReacVol'].iloc[i] = 10.0
    # ##### Assinging peak identification numbers and volumes for each respirogram ######
    for i in range(1, len(Filtered)):
        if Filtered['PeakNo'].iloc[i] == 0:
            continue
        elif Filtered['PeakNo'].iloc[i - 1] == 0:
            Filtered['PeakNo'].iloc[i] = 1
            Filtered['ReacVol'].iloc[i] = 10 + Cal_Volume
        elif Filtered['PeakNo'].iloc[i - 1] == 1:
            Filtered['PeakNo'].iloc[i] = 2
            Filtered['ReacVol'].iloc[i] = 10 + 2 * Cal_Volume
        else:
            Filtered['PeakNo'].iloc[i] = Filtered['PeakNo'].iloc[i - 1] + 1
            Filtered['ReacVol'].iloc[i] = Filtered['ReacVol'].iloc[i - 1] + Sample_Volume
    # #### Calculate OURex #####
    DF['OURex'] = DF['Kla'] * (DF['Ce'] - DF['DO_A']) - DF['DOdt_A']

    # #### Calculate the suface of the respirograms #####
    # Create the needed columns in the Filtered DataFrame
    Columns = ['Area', 'stBOD', 'Mass', 'intdt']
    for name in Columns:
        if name not in Filtered.columns:
            Filtered.insert(len(Filtered.columns), name, np.nan)
    # Calculate the difference between the equilibrium DO concentration and the actual Do concentration throughout the Data
    DF['Deficit'] = DF['Ce'] - DF['DO_A']

    for i in range(len(Filtered)):
        Si = Filtered['Start'][i]
        Ei = Filtered['End'][i]

        # Calculate the Area of each respirogram, and integrate the DO concentration gradient throughout the respirogram (should be near zero)
        Filtered['Area'].iloc[i] = scipy.integrate.trapz(DF['Deficit'][Si:Ei].values, DF['Deficit'][Si:Ei].index.astype(np.int64) / 10**9)
        Filtered['intdt'].iloc[i] = scipy.integrate.trapz(DF['DOdt_A'][Si:Ei].values / 3600, DF['DOdt_A'][Si:Ei].index.astype(np.int64) / 10**9)

        # Calculate the stBOD mass indicated by each respirogram, and convert it to a sample concentration according to the volume of the sample
        Filtered['Mass'].iloc[i] = (-Filtered['intdt'].iloc[i] + Filtered['Area'].iloc[i] * Filtered['Kla'].iloc[i] / 3600) * Filtered['ReacVol'].iloc[i]

        # Return different stBOD concentrations for the respirogram's associated sample according their volume (which changes with the peak identification number)
        if Filtered['PeakNo'].iloc[i] == 0:
            Filtered['stBOD'].iloc[i] = np.nan
        elif (Filtered['PeakNo'].iloc[i] == 1) | (Filtered['PeakNo'].iloc[i] == 2):
            Filtered['stBOD'].iloc[i] = np.nan  # Filtered['Mass'].iloc[i]/Cal_Volume
        else:
            Filtered['stBOD'].iloc[i] = Filtered['Mass'].iloc[i] / Sample_Volume

    # Assign stBOD measurements to the main DataFrame
    stBODResults = Filtered.loc[Filtered['PeakNo'] >= 3.0]
    stBODResults = stBODResults.reset_index(drop=True)
    # stBODResults = Filtered
    # stBODResults = stBODResults.reset_index(drop=True)
    length = len(stBODResults) - 1

    for i in range(1, length):
        DF['stBOD'][stBODResults['Start'][i]:stBODResults['Start'][i + 1]] = stBODResults['stBOD'][i]
    DF['stBOD'][stBODResults['Start'][length]:stBODResults['End'][length]] = stBODResults['stBOD'][length]

    # Plot the resulting stBOD measurements #####

    Label1 = ['DO', 'Ce']
    Label2 = ['stBOD']
    Units1 = ['mg/l', 'mg/l']
    Units2 = ['mg/l']
    marks = ['lines', 'lines']
    start_plot = DF.first_valid_index()
    end_plot = DF.last_valid_index()
    figure = Plotit(DF, start_plot, end_plot, ['DO_A', 'Ce'], ['stBOD'], Label1, Label2, Units1, Units2, marks)
    py.offline.iplot(figure)
    # Select the dataframe rows to return to the user in the Results DataFrame (with the relevant data for each respirogram)
    Return_stBOD = ['Start', 'End', 'Ce', 'Kla', 'stBOD', 'PeakNo']
    Return_DF = ['DO', 'Temp', 'DOdt_A', 'DOdt_smooth', 'DO_A', 'DO_smooth', 'seconds', 'Ce', 'Kla', 'stBOD', 'OURex', 'PeakNo', 'ReacVol']
    return DF.loc[:, Return_DF], stBODResults.loc[:, Return_stBOD]
