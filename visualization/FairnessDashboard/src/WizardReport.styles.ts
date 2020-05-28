import { getTheme, IProcessedStyleSet, IStyle, mergeStyleSets, FontWeights } from "office-ui-fabric-react";

export interface IWizardReportStyles {
    spinner: IStyle;
    header: IStyle;
    multimodelButton: IStyle;
    headerTitle: IStyle;
    headerBanner: IStyle;
    bannerWrapper: IStyle;
    //editButton: IStyle;
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
            margin: "auto",
            padding: "40px"
        },
        header: {
            padding: "0 90px",
            backgroundColor: theme.semanticColors.bodyStandoutBackground//"#F2F2F2"
        },
        multimodelButton: {
            marginTop: "20px",
            padding: 0,
            //color: "#333333",
            //fontSize: "12px",
            //lineHeight: "16px",
            //fontWeight: "400"
        },
        headerTitle: {
            paddingTop: "10px",
            color: theme.semanticColors.bodyText, // "#333333",
            //fontSize: "32px",
            //lineHeight: "39px",
            fontWeight: FontWeights.light //"100"
        },
        headerBanner: {
            display: "flex"
        },
        bannerWrapper: {
            width: "100%",
            paddingTop: "18px",
            paddingBottom: "15px",
            display: "inline-flex",
            flexDirection: "row",
            justifyContent: "space-between"
        },
        // editButton: {
        //     color: "#333333",
        //     fontSize: "12px",
        //     lineHeight: "20px",
        //     fontWeight: "400"
        // },
        metricText: {
            color: theme.semanticColors.bodyText, //"#333333",
            //fontSize: "36px",
            //lineHeight: "44px",
            fontWeight: FontWeights.light, //"100",
            paddingRight: "12px"
        },
        firstMetricLabel: {
            color: theme.semanticColors.bodyText, //"#333333",
            //fontSize: "12px",
            //lineHeight: "16px",
            //fontWeight: "400",
            padding: "8px 12px 0 12px",
            maxWidth: "120px",
            borderRight: "1px solid",// #CCCCCC",
            borderRightColor: theme.semanticColors.bodyDivider,
            marginRight: "20px"
        },
        metricLabel: {
            color: theme.semanticColors.bodyText, //"#333333",
            //fontSize: "12px",
            //lineHeight: "16px",
            //fontWeight: "400",
            paddingTop: "8px",
            maxWidth: "130px"
        },
        presentationArea: {
            display: "flex",
            flexDirection: "row",
            padding: "20px 0 30px 90px",
            //backgroundColor: 'white'
            backgroundColor: theme.semanticColors.bodyBackground
        },
        chartWrapper: {
            flex: "1 0 40%",
            display: "flex",
            flexDirection: "column"
        },
        chartBody: {
            flex: 1
        },
        chartHeader: {
            height: "23px",
            paddingLeft: "10px",
            color: theme.semanticColors.bodyText, //"#333333",
            //fontSize: "12px",
            //lineHeight: "12px",
            //fontWeight: "500"
        },
        mainRight: {
            minWidth: "200px",
            paddingLeft: "35px",
            flexBasis: "300px",
            flexShrink: 1
        },
        rightTitle: {
            color: theme.semanticColors.bodyText, //"#333333",
            //fontSize: "12px",
            //lineHeight: "16px",
            //fontWeight: "500",
            paddingBottom: "11px",
            borderBottom: "1px solid",// #CCCCCC",
            borderBottomColor: theme.semanticColors.bodyDivider
        },
        rightText: {
            padding: "16px 15px 30px 0",
            color: theme.semanticColors.bodyText, //"#333333",
            //fontSize: "15px",
            //lineHeight: "18px",
            //fontWeight: "400",
            borderBottom: "0.5px dashed",// #CCCCCC"
            borderBottomColor: theme.semanticColors.bodyDivider
        },
        insights: {
            textTransform: "uppercase",
            color: theme.semanticColors.bodyText, //"#333333",
            //fontSize: "15px",
            //lineHeight: "16px",
            //fontWeight: "500",
            padding: "18px 0",
        },
        insightsText: {
            color: theme.semanticColors.bodyText, //"#333333",
            //fontSize: "15px",
            //lineHeight: "16px",
            //fontWeight: "400",
            paddingBottom: "18px",
            paddingRight: "15px",
            borderBottom: "1px solid",
            borderBottomColor: theme.semanticColors.bodyDivider// #CCCCCC"
        },
        tableWrapper: {
            paddingBottom: "20px"
        },
        textRow: {
            display: "flex",
            flexDirection: "row",
            alignItems: "center",
            paddingBottom: "7px"
        },
        colorBlock: {
            width: "15px",
            height: "15px",
            marginRight: "9px"
        },
        multimodelSection: {
            display: "flex",
            flexDirection:"row"
        },
        modelLabel: {
            alignSelf: "center",
            paddingLeft: "35px",
            paddingTop: "16px",
            color: theme.semanticColors.bodyText, //"#333333",
            //fontSize: "26px",
            //lineHeight: "16px",
            //fontWeight: "400"
        },
        groupLabel: {
            color: theme.semanticColors.bodyText
        }
    });
};