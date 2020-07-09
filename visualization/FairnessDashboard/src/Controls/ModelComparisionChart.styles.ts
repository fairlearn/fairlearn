import {
    getTheme,
    IProcessedStyleSet,
    IStyle,
    mergeStyleSets,
    FontWeights,
    ThemeSettingName,
} from 'office-ui-fabric-react';

export interface IModelComparisionChartStyles {
    frame: IStyle;
    spinner: IStyle;
    header: IStyle;
    headerTitle: IStyle;
    editButton: IStyle;
    main: IStyle;
    mainRight: IStyle;
    rightTitle: IStyle;
    rightText: IStyle;
    insights: IStyle;
    insightsText: IStyle;
    chart: IStyle;
    textSection: IStyle;
    radio: IStyle;
    radioOptions: IStyle;
}

export const ModelComparisionChartStyles: () => IProcessedStyleSet<IModelComparisionChartStyles> = () => {
    const theme = getTheme();
    return mergeStyleSets<IModelComparisionChartStyles>({
        frame: {
            flex: 1,
            display: 'flex',
            flexDirection: 'column',
        },
        spinner: {
            margin: 'auto',
            padding: '40px',
        },
        header: {
            backgroundColor: theme.semanticColors.bodyFrameBackground,
            padding: '0 90px',
            height: '90px',
            display: 'inline-flex',
            flexDirection: 'row',
            justifyContent: 'space-between',
            alignItems: 'center',
        },
        headerTitle: {
            color: theme.semanticColors.bodyText,
            fontWeight: FontWeights.semibold,
        },
        editButton: {
            color: theme.semanticColors.buttonText,
        },
        main: {
            height: '100%',
            flex: 1,
            display: 'inline-flex',
            flexDirection: 'row',
            backgroundColor: theme.semanticColors.bodyBackground,
        },
        mainRight: {
            padding: '30px 0 0 35px',
            width: '300px',
        },
        rightTitle: {
            color: theme.semanticColors.bodyText,
            paddingBottom: '18px',
            borderBottom: '1px solid',
            borderColor: theme.semanticColors.bodyDivider,
        },
        rightText: {
            padding: '16px 15px 30px 0',
            color: theme.semanticColors.bodyText,
            borderBottom: '0.5px dashed',
            borderColor: theme.semanticColors.bodyDivider,
        },
        insights: {
            textTransform: 'uppercase',
            color: theme.semanticColors.bodyText,
            padding: '18px 0',
        },
        insightsText: {
            paddingBottom: '18px',
            paddingRight: '15px',
            borderBottom: '1px solid',
            borderColor: theme.semanticColors.bodyDivider,
        },
        chart: {
            padding: '60px 0 0 0',
            flex: 1,
        },
        textSection: {
            color: theme.semanticColors.bodyText,
            paddingBottom: '5px',
        },
        radio: {
            paddingBottom: '30px',
            paddingLeft: '75px',
            backgroundColor: theme.semanticColors.bodyBackground,
        },
        radioOptions: {
            color: theme.semanticColors.bodyText,
        },
    });
};
