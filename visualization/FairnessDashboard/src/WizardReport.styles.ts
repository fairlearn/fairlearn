import { getTheme, IProcessedStyleSet, IStyle, mergeStyleSets, FontWeights } from "office-ui-fabric-react";

export interface IWizardReportStyles {
    spinner: IStyle;
    header: IStyle;
    multimodelButton: IStyle;
    headerTitle: IStyle;
    headerBanner: IStyle;
    headerOptions: IStyle;
    bannerWrapper: IStyle;
    metricText: IStyle;
    firstMetricLabel: IStyle;
    metricLabel: IStyle;
    expandAttributes: IStyle;
    overallArea: IStyle;
    presentationArea: IStyle;
    chartWrapper: IStyle;
    chartBody: IStyle;
    chartHeader: IStyle;
    dropDown: IStyle;
    main: IStyle;
    mainLeft: IStyle;
    mainRight: IStyle;
    rightTitle: IStyle;
    rightText: IStyle;
    insights: IStyle;
    insightsText: IStyle;
    downloadReport: IStyle;
    tableWrapper: IStyle;
    textRow: IStyle;
    infoButton: IStyle;
    doneButton: IStyle;
    closeButton: IStyle;
    equalizedOdds: IStyle;
    howTo: IStyle;
    colorBlock: IStyle;
    multimodelSection: IStyle;
    modelLabel: IStyle;
    modalContentHelp: IStyle;
    groupLabel: IStyle;
    legendPanel: IStyle;
    legendTitle: IStyle;
    legendSubtitle: IStyle;
}

export const WizardReportStyles: () => IProcessedStyleSet<IWizardReportStyles> = () => {
    const theme = getTheme();
    return mergeStyleSets<IWizardReportStyles>({
        spinner: {
            margin: "auto",
            padding: "40px"
        },
        header: {
            padding: "0 100px",
            marginBottom: "20px",
            backgroundColor: theme.semanticColors.bodyStandoutBackground
        },
        multimodelButton: {
            marginTop: "20px",
            padding: 0,
        },
        headerTitle: {
            paddingTop: "10px",
            color: theme.semanticColors.bodyText
        },
        headerBanner: {
            display: "flex"
        },
        headerOptions: {
            backgroundColor: theme.semanticColors.bodyStandoutBackground,
        },
        bannerWrapper: {
            width: "100%",
            paddingTop: "18px",
            paddingBottom: "15px",
            display: "inline-flex",
            flexDirection: "row",
            justifyContent: "space-between"
        },
        metricText: {
            color: theme.semanticColors.bodyText,
            paddingRight: "12px",
            fontWeight: FontWeights.light
        },
        firstMetricLabel: {
            color: theme.semanticColors.bodyText,
            padding: "8px 12px 0 12px",
            maxWidth: "120px",
            borderRight: "1px solid",
            borderRightColor: theme.semanticColors.bodyDivider,
            marginRight: "20px"
        },
        metricLabel: {
            color: theme.semanticColors.bodyText,
            paddingTop: "8px",
            maxWidth: "130px"
        },
        expandAttributes: {
            color: "#333333",
            fontSize: "12px",
            lineHeight: "16px",
            fontWeight: "normal",
            height: "26px",
            marginLeft: "100px",
            marginBottom: "20px"
        },
        overallArea: {
            display: "flex",
            flexDirection: "row",
            padding: "20px 0 30px 100px",
            backgroundColor: 'white'
        },
        presentationArea: {
            display: "flex",
            flexDirection: "row",
            padding: "20px 0 30px 90px",
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
            color: theme.semanticColors.bodyText
        },
        main: {
            display: "flex",
            flexDirection: "row"
        },
        mainLeft: {
            width: "75%",
        },
        dropDown: {
            margin: "10px 0px",
            display: "inline-block"
        },
        mainRight: {
            minWidth: "200px",
            paddingLeft: "35px",
            flexBasis: "300px",
            flexShrink: 1
        },
        rightTitle: {
            color: theme.semanticColors.bodyText,
            paddingBottom: "11px",
            borderBottom: "1px solid",
            borderBottomColor: theme.semanticColors.bodyDivider
        },
        rightText: {
            padding: "16px 15px 30px 0",
            color: theme.semanticColors.bodyText,
            borderBottom: "0.5px dashed",
            borderBottomColor: theme.semanticColors.bodyDivider
        },
        insights: {
            textTransform: "uppercase",
            color: theme.semanticColors.bodyText,
            padding: "18px 0",
            display: "inline"
        },
        insightsText: {
            marginTop: "20px",
            color: theme.semanticColors.bodyText,
            paddingBottom: "18px",
            paddingRight: "15px",
            borderBottom: "1px solid",
            borderBottomColor: theme.semanticColors.bodyDivider
        },
        downloadReport: {
            color: "#333333",
            fontSize: "12px",
            lineHeight: "16px",
            fontWeight: "normal",
            paddingTop: "20px",
            paddingBottom: "20px",
            paddingLeft: "0px",
        },
        tableWrapper: {
            paddingBottom: "20px"
        },
        textRow: {
            display: "flex",
            flexDirection: "row",
            alignItems: "center",
            paddingBottom: "7px",
            color: theme.semanticColors.bodyText
        },
        infoButton: {
            color: "#999999",
            float: "left",
            width: "15px",
            height: "15px",
            textAlign: "center",
            fontSize: "12px",
            lineHeight: "14px",
            fontWeight: "600",
            borderRadius: "50%",
            border: "1px solid",
            marginTop: "3px",
            marginRight: "3px",
            marginLeft: "250px"
        },
        closeButton: {
            color: "#FFFFFF",
            float: "right",
            fontFamily: "Arial",
            fontSize: "20px",
            lineHeight: "20px",
            fontWeight: "400",
            paddingLeft: "20px"
        },
        doneButton: {
            color: "#FFFFFF",
            borderRadius: "5px",
            background: "#5A53FF",
            padding: "5px 15px",
            selectors: {
                '&:hover': { color: "#ffffff" }
            }
        },
        equalizedOdds: {
            float: "left",
            fontSize: "18px",
            lineHeight: "22px",
            fontWeight: "normal",
            paddingTop: "30px",
            paddingLeft: "100px"
        },
        howTo: {
            paddingTop: "20px",
            paddingLeft: "100px"
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
        modalContentHelp: {
            float: 'left',
            paddingTop: '10px',
            paddingRight: '20px',
            wordWrap: "break-word",
            width: "300px",
            textAlign: "center"
        },
        modelLabel: {
            fontSize: "24px",
            alignSelf: "center",
            paddingTop: "16px",
            color: theme.semanticColors.bodyText
        },
        groupLabel: {
            color: theme.semanticColors.bodyText
        },
        legendPanel: {
            marginLeft: "100px"
        },
        legendTitle: {
            color: "#333333",
            fontSize: "12px",
            lineHeight: "16px",
        },
        legendSubtitle: {
            color: "#666666",
            fontSize: "9px",
            lineHeight: "12x",
            fontStyle: "italic"
        }
    });
};