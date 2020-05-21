import { getTheme, IProcessedStyleSet, IStyle, mergeStyleSets } from "office-ui-fabric-react";

export interface IFeatureTabStyles {
    itemCell: IStyle;
    iconClass: IStyle;
    itemsList: IStyle;
    frame: IStyle;
    main: IStyle;
    header: IStyle;
    textBody: IStyle;
    tableHeader: IStyle;
    itemTitle: IStyle;
    valueCount: IStyle;
    iconWrapper: IStyle;
    featureDescriptionSection: IStyle;
    binSection: IStyle;
    expandButton: IStyle;
    category: IStyle;
    subgroupHeader: IStyle;

}

export const FeatureTabStyles: () => IProcessedStyleSet<IFeatureTabStyles> = () => {
    const theme = getTheme();
    return mergeStyleSets<IFeatureTabStyles>({
        itemCell: {
            display: "flex",
            flexDirection: "row",
            padding: "20px 0",
            width: "100%",
            cursor: "pointer",
            boxSizing: "border-box",
            borderBottom: "1px solid",
            borderColor: "#CCCCCC",
            selectors: {
              '&:hover': { background: "light grey" }
            }
        },
        iconClass: {
            fontSize: "20px"
        },
        itemsList: {
            overflowY: "auto"
        },
        frame: {
            height: "100%",
        },
        main: {
            height: "100%",
            maxWidth: "700px",
            flex: 1
        },
        header: {
            color: "#333333",
            fontSize: "32px",
            lineHeight: "39px",
            fontWeight: "100"
        },
        textBody: {
            paddingTop: "12px",
            fontSize: "18px",
            lineHeight: "24px",
            fontWeight: "300",
            marginBottom: "15px"
        },
        tableHeader: {
            display: "flex",
            flexDirection: "row",
            justifyContent: "space-between",
            paddingBottom: "15px",
            color: "#333333",
            fontSize: "15px",
            lineHeight: "18px",
            fontWeight: "500",
            borderBottom: "1px solid",
            borderColor: "#CCCCCC"

        },
        itemTitle: {
            margin: 0,
            color: "#333333",
            fontSize: "22px",
            lineHeight: "26px",
            fontWeight: "300"
        },
        valueCount: {
            paddingTop: "15px",
            color: "#333333",
            fontSize: "15px",
            lineHeight: "18px",
            fontWeight: "500"
        },
        iconWrapper: {
            paddingTop: "4px",
            paddingLeft: "5px",
            width: "30px"
        },
        featureDescriptionSection: {
            flex: 1,
            paddingRight: "20px",
            minHeight:"75px"
        },
        binSection:{
            width:"130px",

        },
        expandButton: {
            paddingLeft: 0,
            selectors: {
                "& i":{
                    marginLeft: 0
                }
            }
        },
        category: {
            textOverflow: "ellipsis",
            whiteSpace: "nowrap",
            overflow: "hidden"
        },
        subgroupHeader: {
            width: "130px"
        }
    });
};