import { getTheme, IProcessedStyleSet, IStyle, mergeStyleSets, FontWeights } from 'office-ui-fabric-react';

export interface IWizardReportStyles {
    spinner: IStyle;
    header: IStyle;
    multimodelButton: IStyle;
    headerTitle: IStyle;
    headerBanner: IStyle;
    bannerWrapper: IStyle;
    metricText: IStyle;
    firstMetricLabel: IStyle;
    metricLabel: IStyle;
    presentationArea: IStyle;
    chartWrapper: IStyle;
    chartBody: IStyle;
    chartHeader: IStyle;
    mainRight: IStyle;
    rightTitle: IStyle;
    rightText: IStyle;
    insights: IStyle;
    insightsText: IStyle;
    tableWrapper: IStyle;
    textRow: IStyle;
    colorBlock: IStyle;
    multimodelSection: IStyle;
    modelLabel: IStyle;
    groupLabel: IStyle;
}

export const WizardReportStyles: () => IProcessedStyleSet<IWizardReportStyles> = () => {
    const theme = getTheme();
    return mergeStyleSets<IWizardReportStyles>({
        spinner: {
            margin: 'auto',
            padding: '40px',
        },
        header: {
            padding: '0 90px',
            backgroundColor: theme.semanticColors.bodyStandoutBackground,
        },
        multimodelButton: {
            marginTop: '20px',
            padding: 0,
        },
        headerTitle: {
            paddingTop: '10px',
            color: theme.semanticColors.bodyText,
        },
        headerBanner: {
            display: 'flex',
        },
        bannerWrapper: {
            width: '100%',
            paddingTop: '18px',
            paddingBottom: '15px',
            display: 'inline-flex',
            flexDirection: 'row',
            justifyContent: 'space-between',
        },
        metricText: {
            color: theme.semanticColors.bodyText,
            paddingRight: '12px',
            fontWeight: FontWeights.light,
            lineHeight: '44px',
            fontSize: '36px',
        },
        firstMetricLabel: {
            color: theme.semanticColors.bodyText,
            padding: '8px 12px 0 12px',
            maxWidth: '120px',
            borderRight: '1px solid',
            borderRightColor: theme.semanticColors.bodyDivider,
            marginRight: '20px',
            lineHeight: '16px',
        },
        metricLabel: {
            color: theme.semanticColors.bodyText,
            paddingTop: '8px',
            maxWidth: '130px',
        },
        presentationArea: {
            display: 'flex',
            flexDirection: 'row',
            padding: '20px 0 30px 90px',
            backgroundColor: theme.semanticColors.bodyBackground,
        },
        chartWrapper: {
            flex: '1 0 40%',
            display: 'flex',
            flexDirection: 'column',
        },
        chartBody: {
            flex: 1,
        },
        chartHeader: {
            height: '23px',
            paddingLeft: '10px',
            color: theme.semanticColors.bodyText,
        },
        mainRight: {
            minWidth: '200px',
            paddingLeft: '35px',
            flexBasis: '300px',
            flexShrink: 1,
        },
        rightTitle: {
            color: theme.semanticColors.bodyText,
            paddingBottom: '11px',
            borderBottom: '1px solid',
            borderBottomColor: theme.semanticColors.bodyDivider,
        },
        rightText: {
            padding: '16px 15px 30px 0',
            color: theme.semanticColors.bodyText,
            borderBottom: '0.5px dashed',
            borderBottomColor: theme.semanticColors.bodyDivider,
        },
        insights: {
            textTransform: 'uppercase',
            color: theme.semanticColors.bodyText,
            padding: '18px 0',
        },
        insightsText: {
            color: theme.semanticColors.bodyText,
            paddingBottom: '18px',
            paddingRight: '15px',
            borderBottom: '1px solid',
            borderBottomColor: theme.semanticColors.bodyDivider,
        },
        tableWrapper: {
            paddingBottom: '20px',
        },
        textRow: {
            display: 'flex',
            flexDirection: 'row',
            alignItems: 'center',
            paddingBottom: '7px',
            color: theme.semanticColors.bodyText,
        },
        colorBlock: {
            width: '15px',
            height: '15px',
            marginRight: '9px',
        },
        multimodelSection: {
            display: 'flex',
            flexDirection: 'row',
        },
        modelLabel: {
            alignSelf: 'center',
            paddingLeft: '35px',
            paddingTop: '16px',
            color: theme.semanticColors.bodyText,
        },
        groupLabel: {
            color: theme.semanticColors.bodyText,
        },
    });
};
